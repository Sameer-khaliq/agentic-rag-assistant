from pathlib import Path
import os
import sys
import asyncio

# Ensure inside nested operations that the root path can load 'src' cleanly
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

# ---------------------------------------------------------------------------
# Custom Styling & Theme Configuration (Hidden Timer + Modern Palette)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* Hide any runtime generation timers, ETA, and progress counter texts */
.progress-text, 
.eta, 
.timer, 
.statustracker, 
[data-testid="status-tracker"], 
.generating, 
.meta-text {
    display: none !important;
}

footer {
    display: none !important;
}

/* Container Styling */
.gradio-container {
    max-width: 950px !important;
    margin: 0 auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* Header Banner */
.custom-header {
    text-align: center;
    padding: 1.5rem 1rem;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.12) 0%, rgba(99, 102, 241, 0.12) 100%);
    border-radius: 16px;
    border: 1px solid rgba(99, 102, 241, 0.2);
}

.custom-header h1 {
    font-size: 1.85rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
    background: linear-gradient(135deg, #06b6d4 0%, #6366f1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.custom-header p {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 0;
}

/* Chatbot Area */
.chatbot-wrap {
    border-radius: 16px !important;
    overflow: hidden !important;
}

/* Input area */
textarea:focus {
    border-color: #06b6d4 !important;
    box-shadow: 0 0 0 2px rgba(6, 182, 212, 0.25) !important;
}
"""

CUSTOM_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.cyan,
    secondary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "Segoe UI", "sans-serif"],
).set(
    body_background_fill="*neutral_950",
    body_background_fill_dark="*neutral_950",
    block_background_fill="*neutral_900",
    block_background_fill_dark="*neutral_900",
    block_border_width="1px",
    block_border_color="*neutral_800",
    panel_border_color="*neutral_800",
    button_primary_background_fill="linear-gradient(135deg, #06b6d4 0%, #6366f1 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #0891b2 0%, #4f46e5 100%)",
    button_primary_text_color="#ffffff",
)


# ---------------------------------------------------------------------------
# Async Generator Handler with Phased Thinking Words
# ---------------------------------------------------------------------------
async def respond(message: str, history: list):
    """
    Async streaming handler that displays progressive thinking status updates
    before presenting the final synthesized response.
    """
    # Phase 1: 1s Delay
    yield " *Processing your input...*"
    await asyncio.sleep(1.0)

    # Phase 2: 2s Delay
    yield " *Synthesizing answer...*"
    await asyncio.sleep(2.0)

    # Phase 3: 3s Delay
    yield " *Presenting you the reasonable answer...*"
    await asyncio.sleep(3.0)

    # Phase 4: Final Output Execution
    try:
        answer = await ask_agent_async(message)
    except Exception as e:
        logger.error(f"UI routing catch error encountered: {str(e)}", exc_info=True)
        answer = "Agent Services unavailable at the moment try again later"

    # Thinking words disappear and are completely replaced by final answer
    yield answer


# ---------------------------------------------------------------------------
# Gradio Chat Interface Definition (Gradio 6.x Compatible)
# ---------------------------------------------------------------------------
with gr.Blocks(
    theme=CUSTOM_THEME,
    css=CUSTOM_CSS,
    title="🤖 Agentic RAG Assistant",
) as demo:
    gr.ChatInterface(
        fn=respond,
        title="🤖 Agentic RAG Assistant",
        description=(
            "Autonomous ReAct Assistant powered by Groq (GPT-OSS 120B/20B) with "
            "Document Retrieval, AST Math, and Real-Time Web Search."
        ),
        examples=[
            "What is a centralized database?",
            "What are the types of computers based on size?",
            "What is 15 percent of 2400?",
            "Who is the current CEO of OpenAI?",
            "Write a full-stack e-commerce system using Django and Next.js",
        ],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)