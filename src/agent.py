from __future__ import annotations

import re
import asyncio
import time
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic.agents.agent import AgentOutputParser, AgentAction, AgentFinish
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from src.tools import build_tools
from src.config import settings
from src.logger import get_logger
from src.gating import run_prefilter

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# #7 — Tighter REACT_PROMPT (shorter phrasing, same behavior, fewer prefill tokens)
# ---------------------------------------------------------------------------
REACT_PROMPT = """You are an Agentic RAG Assistant with three tools: local knowledge base retrieval, calculator, and web search.

Tools available:
{tools}

Rules:
- Databases/computers/computer-types: ALWAYS use KnowledgeBaseRetriever first, even if you know the answer.
- Out-of-scope (apps, code generation, hacking): skip tools, respond with scope limitation.
- Format labels in plain text only — no markdown bolding (**).
- Output ONE step at a time. Stop after Action Input. Never write Observation yourself.

Format:
Thought: <your reasoning>
Action: <one of [{tool_names}]>
Action Input: <input>
Observation: <tool result>
... repeat as needed ...
Thought: I now know the final answer
Final Answer: <answer>

Question: {input}
Thought:{agent_scratchpad}"""


# ---------------------------------------------------------------------------
# Robust parser — handles modern LLM markdown bolding + one-shot final answers
# ---------------------------------------------------------------------------
class RobustReActOutputParser(AgentOutputParser):
    """
    Robust ReAct parser for modern chat models like gpt-oss-120b.
    - Strips markdown bolding (**Thought:** -> Thought:, **Action:** -> Action:).
    - Extracts Final Answer immediately if present.
    - Never raises unhandled parser exceptions that cause infinite retry loops.
    """

    def parse(self, text: str) -> AgentAction | AgentFinish:
        cleaned = re.sub(r"\*\*([A-Za-z\s]+):\*\*", r"\1:", text)
        cleaned = re.sub(r"\*\*([A-Za-z\s]+)\*\*\s*:", r"\1:", cleaned)

        # 1. Final Answer present — accept immediately
        if "Final Answer:" in cleaned:
            final_content = cleaned.rsplit("Final Answer:", 1)[-1].strip()
            return AgentFinish({"output": final_content}, text)

        # 2. Action + Action Input present — extract cleanly
        action_match = re.search(
            r"Action:\s*(.*?)\n\s*Action Input:\s*(.*)", cleaned, re.DOTALL
        )
        if action_match:
            action = action_match.group(1).strip().strip("`").strip('"').strip("'")
            raw_input = action_match.group(2)
            raw_input = re.split(r"\nObservation\s*:?", raw_input)[0].strip()
            tool_input = raw_input.strip('"').strip("'")
            return AgentAction(action, tool_input, text)

        # 3. Thought without Action — treat as final answer
        if "Thought:" in cleaned and "Action:" not in cleaned:
            thought_content = cleaned.split("Thought:", 1)[-1].strip()
            return AgentFinish({"output": thought_content}, text)

        # 4. Graceful fallback
        return AgentFinish({"output": cleaned.strip()}, text)

    @property
    def _type(self) -> str:
        return "robust-react"


_cached_executor: AgentExecutor | None = None


def build_agent(return_intermediate_steps: bool = False) -> AgentExecutor:
    logger.info(f"Constructing Groq ReAct execution engine with {settings.GROQ_AGENT_MODEL}...")

    llm = ChatGroq(
        model=settings.GROQ_AGENT_MODEL,
        groq_api_key=settings.GROQ_API_KEY,
        temperature=0.1,
        max_retries=2,       # #3 — was 3 (3rd retry = 24s+ wasted)
        timeout=8.0,         # #3 — was 30s (actual worst case ~2s, 8s = generous)
        streaming=True,      # #5 — stream tokens to client for perceived latency
    )

    tools = build_tools()
    prompt = PromptTemplate.from_template(REACT_PROMPT)
    agent = create_react_agent(llm, tools, prompt, output_parser=RobustReActOutputParser())

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=settings.DEBUG,          # #2 — off in prod, on when DEBUG=True in .env
        handle_parsing_errors=True,
        max_iterations=settings.AGENT_MAX_ITERATIONS,  # #6 — configurable via env (default=4)
        return_intermediate_steps=return_intermediate_steps,
    )
    return executor


def get_agent_executor(return_intermediate_steps: bool = False) -> AgentExecutor:
    global _cached_executor
    if _cached_executor is None or return_intermediate_steps:
        executor = build_agent(return_intermediate_steps=return_intermediate_steps)
        if not return_intermediate_steps:
            _cached_executor = executor
        return executor
    return _cached_executor


DEGRADATION_FALLBACK_MESSAGE = "Agent Services unavailable at the moment try again later"


# ---------------------------------------------------------------------------
# #1 — Async ask_agent (ainvoke) + sync wrapper for Gradio compatibility
# ---------------------------------------------------------------------------
async def ask_agent_async(query: str) -> str:
    """
    Async entrypoint. Allows concurrent queries without blocking the event loop.
    Gradio/FastAPI can await this directly for full async throughput.
    """
    logger.info(f"Agent router receiving query: {query}")
    t0 = time.perf_counter()

    # Layer 0 — Fast prefilter (regex, ~0.1ms, 0 API calls)
    gate_result = run_prefilter(query)
    if gate_result and gate_result.get("gated"):
        logger.info(
            f"Prefiltered [{gate_result.get('category')}] in "
            f"{(time.perf_counter()-t0)*1000:.1f}ms"
        )
        return gate_result["response"]

    try:
        executor = get_agent_executor()
        result = await executor.ainvoke({"input": query})   # #1 — async invoke
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"Agent completed in {elapsed:.0f}ms")
        return result["output"]
    except Exception as e:
        logger.error(f"Agent execution failure: {str(e)}", exc_info=True)
        return DEGRADATION_FALLBACK_MESSAGE


def ask_agent(query: str) -> str:
    """
    Sync wrapper over ask_agent_async — used by Gradio's synchronous fn= interface.
    Runs the async function in the current or a new event loop safely.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Inside an existing event loop (e.g. Jupyter / some ASGI contexts)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, ask_agent_async(query))
                return future.result()
        else:
            return loop.run_until_complete(ask_agent_async(query))
    except Exception as e:
        logger.error(f"Sync wrapper failure: {str(e)}", exc_info=True)
        return DEGRADATION_FALLBACK_MESSAGE


if __name__ == "__main__":
    try:
        answer = ask_agent("What is a centralized database?")
        print(f"\n[TEST]: {answer}")
    except Exception:
        pass