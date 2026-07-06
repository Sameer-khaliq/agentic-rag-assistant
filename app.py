import os
import sys

# Ensure inside nested operations that the root path can load 'src' clean
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from src.logger import get_logger
from src.ingest import build_vector_store
from src.agent import ask_agent

logger = get_logger(__name__)

# System validation: Startup runtime extraction
if not os.path.exists("chroma_db"):
    logger.info("Chroma vector store directory missing. Initializing database extraction framework...")
    build_vector_store()
else:
    logger.info("Chroma vector store directory located. Skipping raw data initialization.")

def respond(message, history):
    """
    Gradio execution interface that pipes user queries directly to the low-latency 
    Groq ReAct core engine.
    """
    try:
        answer = ask_agent(message)
    except Exception as e:
        logger.error(f"UI routing catch error encountered: {str(e)}")
        answer = (
            "System Error: Unable to complete your request.\n"
            "Please check backend connection mappings and parameters."
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
        "Write a full-stack e-commerce system using Django and Next.js" # This will trigger your shut-up call guardrail!
    ],
    
)

if __name__ == "__main__":
    # Explicit container port mapping binding configuration
    demo.launch(server_name="0.0.0.0", server_port=7860)