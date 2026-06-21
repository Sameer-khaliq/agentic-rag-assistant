# Agentic RAG Assistant 🤖

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)
[![LangChain](https://img.shields.io/badge/LangChain-ReAct%20Agent-green.svg)](https://langchain.com)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/demo-live-success.svg)](https://sameerkhaliq-agentic-rag-assistant.hf.space/)

A ReAct agent that dynamically chooses between **document retrieval**,
**calculation**, and **live web search** based on the query — instead of
always running a fixed retrieval pipeline.

**[Live Interactive Demo →](https://sameerkhaliq-agentic-rag-assistant.hf.space/)**

---

## 🏗️ System Architecture & Workflow

## Architecture

![Architecture diagram](architecture.png)
---

## 📊 Benchmarks

### 1. Retrieval quality (RAGAS evaluation, 20 test queries):

| Metric | Score |
|---|---|
| Faithfulness | 1.00 |
| Answer Relevancy | 0.85 |
| Context Recall | 1.00 |

### 2. Contextual compression (5 test queries):

| Metric | Value |
|---|---|
| Average context size reduction | 85.7% |
| Metadata fields filterable | 2 (category, source) |

### 3. Agentic tool selection (15 diverse queries — retrieval / calculation / web search):

| Metric | Score |
|---|---|
| Correct tool selection | 15/15 (100%) |
| Clean execution (verified via observation-level error checking) | 15/15 (100%) |

---

## 💡 Why agentic over fixed-pipeline RAG?

A fixed RAG pipeline always retrieves, regardless of what the query actually
needs — it can't do math, and it can't answer "what's today's date." This
agent reasons about *which* capability a question needs before acting,
using the [ReAct](https://arxiv.org/abs/2210.03629) (Reason + Act) pattern.

---

## 🛠️ Tech stack

LangChain (ReAct agent) · Gemini 2.5 Flash · ChromaDB · Tavily Search · Gradio · RAGAS (evaluation)

---

## 💻 Run locally

```bash
git clone https://github.com/Sameer-khaliq/agentic-rag-assistant.git
cd agentic-rag-assistant
uv venv
uv pip sync requirements.txt
cp .env.example .env  # add your API keys
uv run app.py