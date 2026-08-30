from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Core modular infrastructure references
from src.tools import build_tools
from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

# System instructions updated with fallbacks and custom boundary checks
REACT_PROMPT = """You are an Advanced Autonomous Assistant. You have access to these specific tools:

{tools}

CRITICAL RULES FOR OUT-OF-SCOPE QUERIES:
If the user asks for something completely out-of-scope, massively complex, or outside your core purpose (e.g., writing full-stack applications, complete system design, hacking, or heavy tasks unrelated to quick math, live search, or your local data), you MUST NOT use any Action. Skip directly to:
Thought: This query is out of scope for my tools.
Final Answer: I am specialized in answering queries from local knowledge base documents (computers and databases), performing mathematical calculations, and conducting real-time web searches. This request is outside the scope of my capabilities.

MANDATORY RETRIEVAL RULE:
If the query is about databases, database types, computers, or computer types — even if you already know the answer — you MUST use the KnowledgeBaseRetriever tool first. Never answer these topics directly from internal knowledge, since the local documents may contain specific details, definitions, or classifications that differ from general knowledge. This rule overrides the general conversational handling below.

GENERAL/CONVERSATIONAL QUERY HANDLING:
If the query is simple, a greeting, or a generic question NOT related to databases or computers, and can be answered directly using your internal knowledge, skip Action entirely and go straight to Final Answer.

Otherwise, for queries that genuinely need a tool, follow the strict ReAct format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

import re
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from src.gating import run_prefilter

class RobustReActOutputParser(ReActSingleInputOutputParser):
    """Normalizes modern markdown bolding (**Action:** -> Action:) so ReAct parsers don't fail."""
    def parse(self, text: str):
        cleaned = re.sub(r"\*\*([A-Za-z\s]+):\*\*", r"\1:", text)
        cleaned = re.sub(r"\*\*([A-Za-z\s]+)\*\*\s*:", r"\1:", cleaned)
        return super().parse(cleaned)

_cached_executor = None


def build_agent(return_intermediate_steps: bool = False) -> AgentExecutor:
    logger.info(f"Constructing Groq ReAct execution engine with {settings.GROQ_AGENT_MODEL}...")
    
    llm = ChatGroq(
        model=settings.GROQ_AGENT_MODEL, 
        groq_api_key=settings.GROQ_API_KEY, 
        temperature=0.1,
        max_retries=3,
        timeout=30.0,
    )
    
    tools = build_tools()
    prompt = PromptTemplate.from_template(REACT_PROMPT)

    agent = create_react_agent(llm, tools, prompt, output_parser=RobustReActOutputParser())
    
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,
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


def ask_agent(query: str) -> str:
    logger.info(f"Agent router receiving query event: {query}")

    # Layer 0/1 Fast Prefilter: Intercept greetings, abuse, credentials, and out-of-scope tasks with zero LLM calls
    gate_result = run_prefilter(query)
    if gate_result and gate_result.get("gated"):
        logger.info(f"Query pre-filtered successfully: {gate_result.get('category')} ({gate_result.get('reason')})")
        return gate_result["response"]

    try:
        executor = get_agent_executor()
        result = executor.invoke({"input": query})
        return result["output"]
    except Exception as e:
        logger.error(f"Critical execution failure inside agent loop: {str(e)}", exc_info=True)
        raise e


if __name__ == "__main__":
    # Internal test execution
    try:
        answer = ask_agent("Hi, who are you?")
        print(f"\n[TEST - GREETING RESPONSE]: {answer}")
    except Exception:
        pass