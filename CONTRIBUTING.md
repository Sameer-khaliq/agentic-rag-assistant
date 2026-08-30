# Contributing to Agentic RAG Assistant

Thanks for your interest in contributing! This document outlines how to set up the project locally and the conventions used in this codebase.

## Project Overview
An agentic RAG system that combines document retrieval (ChromaDB + Gemini embeddings), live web search (Tavily), and mathematical computation (Calculator) through a ReAct agent powered by Groq's GPT-OSS 120B & 20B with prefilter gating.

## Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- API keys: Groq, Tavily, Google Gemini

## Setup

1. Clone the repository
   ```bash
   git clone https://github.com/Sameer-khaliq/agentic-rag-assistant.git
   cd agentic-rag-assistant
   ```

2. Copy the environment template and add your keys
   ```bash
   cp .env.example .env
   ```

3. Install dependencies
   ```bash
   uv sync
   ```

4. Run the app
   ```bash
   uv run python app.py
   ```
   Visit `http://localhost:7860` in your browser.

5. (Optional) Run via Docker
   ```bash
   docker-compose up
   ```

## Project Structure
```
src/
├── config.py    # Pydantic settings, loads .env
├── logger.py    # Centralized JSON logging
├── ingest.py    # Vector store creation pipeline
├── tools.py     # Agent tools (retriever, calculator, web search)
└── agent.py     # ReAct agent construction and prompt logic
app.py           # Gradio UI entrypoint
```

## Development Conventions
- **Functional style preferred** — avoid unnecessary classes; prefer plain functions and Pydantic models for data structures.
- **No hardcoded secrets** — all API keys must be loaded via `src/config.py`'s `Settings` object, never read directly from `os.environ` inside feature code.
- **Logging over print()** — use `get_logger(__name__)` from `src/logger.py` for all runtime output; no bare `print()` statements in `src/`.
- **Error handling** — wrap all external API calls (Groq, Gemini, Tavily, ChromaDB) in try/except with specific exception types; never use bare `except:`.

## Testing Changes
Before submitting changes, verify:
1. `uv run python app.py` starts without errors
2. `docker build -t agentic-rag-assistant .` completes successfully
3. `docker-compose up` runs the service standalone
4. No secrets are present in git history: `git log -p | grep -i "api_key\|secret"`

## Known Limitations
- Embedding calls depend on Gemini free-tier rate limits; heavy concurrent usage may require `time.sleep()` backoff.
- Web search (Tavily) requires outbound network access; will fail in network-restricted environments.