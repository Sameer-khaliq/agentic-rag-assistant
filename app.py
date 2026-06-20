import os
import sys
import gradio as gr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import ask_agent

def respond(message, history):
    """
    Gradio routing function that pipes user input into our ReAct engine.
    """
    try:
        answer = ask_agent(message)
    except Exception as e:
        answer = f"Runtime Error: {e}\nPlease check if API keys and chroma_db are initialized correctly."
    return answer


demo = gr.ChatInterface(
    fn=respond,
    title="Agentic RAG Assistant",
    description=(
        "An advanced Autonomous ReAct Agent that dynamically switches between "
        "Local Document Retrieval (with Contextual Compression), Mathematical Computation, "
        "and Live Web Search (via Tavily) based on your question's intent."
    ),
    examples=[
        "What is a centralized database?",
        "What are the types of computers based on size?",
        "What is 15 percent of 2400?",
        "Who is the current CEO of OpenAI?",
    ],
    theme="soft"
)

if __name__ == "__main__":
    
    demo.launch()