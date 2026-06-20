---
title: Agentic RAG Assistant
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
python_version: 3.11
app_file: app.py
pinned: false
license: mit
short_description: Autonomous ReAct RAG agent with dynamic tool routing.
---
# Agentic RAG Assistant 🤖

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)
[![LangChain](https://img.shields.io/badge/LangChain-ReAct%20Agent-green.svg)](https://langchain.com)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/demo-live-success.svg)](https://huggingface.co/spaces/sameerkhaliq/agentic-rag-assistant)

A ReAct agent that dynamically chooses between **document retrieval**,
**calculation**, and **live web search** based on the query — instead of
always running a fixed retrieval pipeline.

**[Live Interactive Demo →](https://huggingface.co/spaces/sameerkhaliq/agentic-rag-assistant)**

---

## 🏗️ System Architecture & Workflow

<pre><code>
       [ User Question ]
               │
               ▼
     ┌───────────────────┐
     │   ReAct Agent     │◄───────┐
     │ (Reasoning Loop)  │        │ (Inspect Execution)
     └─────────┬─────────┘        │
               │                  │
 ┌─────────────┼─────────────┐    │
 │             │             │    │
 ▼             ▼             ▼    │
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Calculator  │ │  Compressed  │ │  Tavily Web  │
│  Math Tool   │ │  Retriever   │ │  Live Search │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       │                ▼                │
       │         ┌──────────────┐        │
       │         │  ChromaDB    │        │
       │         └──────┬───────┘        │
       │                │                │
       │                ▼                │
       │         ┌──────────────┐        │
       │         │ Contextual   │        │
       │         │ Compression  │        │
       │         └──────┬───────┘        │
       │                │                │
       └───────────────►┼◄───────────────┘
                        │
                        ▼
               [ Synthesized Answer ]
</code></pre>

---

## 📊 Benchmarks

### 1. Retrieval quality (RAGAS evaluation, 20 test queries):

| Metric | Score |
|---|---|
| Faithfulness | 0.80 |
| Answer Relevancy | 0.65 |
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