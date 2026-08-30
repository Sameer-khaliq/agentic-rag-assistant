import sys
from src.agent import ask_agent

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\nQuery: {query}\n" + "-" * 50)
        answer = ask_agent(query)
        print(f"\nResponse:\n{answer}\n")
    else:
        print("Agentic RAG Assistant CLI")
        print("Usage: python main.py <your question>")
        print("Example: python main.py 'What is a centralized database?'")


if __name__ == "__main__":
    main()
