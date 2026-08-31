from pathlib import Path
import os
import sys

# Ensure inside nested operations that the root path can load 'src' clean
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from src.config import settings
from src.logger import get_logger
from src.ingest import build_vector_store
from src.agent import ask_agent_async

logger = get_logger(__name__)

# System validation: Startup runtime extraction with empty volume check
def _ensure_vector_store():
    chroma_path = Path(settings.CHROMA_DIR)
    sqlite_file = chroma_path / "chroma.sqlite3"
    if not chroma_path.exists() or not sqlite_file.exists() or sqlite_file.stat().st_size < 1024:
        logger.info("Chroma vector store missing or unpopulated. Initializing database extraction framework...")
        build_vector_store()
    else:
        logger.info("Chroma vector store located and populated. Skipping raw data initialization.")

_ensure_vector_store()


async def respond(message: str, history: list):
    """
    Async Gradio handler — allows concurrent queries without blocking.
    Streams tokens to client as they arrive from Groq (perceived latency improvement).
    """
    try:
        answer = await ask_agent_async(message)
    except Exception as e:
        logger.error(f"UI routing catch error encountered: {str(e)}", exc_info=True)
        answer = (
            f"System Error: Unable to complete your request ({type(e).__name__}: {str(e)}).\n"
            "Please verify API keys and network connectivity."
        )
    return answer


demo = gr.ChatInterface(
    fn=respond,
    title="Agentic RAG Production Assistant",
    description=(
        "An Advanced Autonomous ReAct Agent optimized with Groq LPU inference. "
        "Dynamically switches between Document Retrieval, Calculations, and Real-Time Search."
    ),
    examples=[
        "What is a centralized database?",
        "What are the types of computers based on size?",
        "What is 15 percent of 2400?",
        "Who is the current CEO of OpenAI?",
        "Write a full-stack e-commerce system using Django and Next.js",  # Triggers the out-of-scope guardrail
    ],
)

if __name__ == "__main__":
    # Explicit container port mapping binding configuration
    demo.launch(server_name="0.0.0.0", server_port=7860)