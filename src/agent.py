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
Final Answer: Main sirf selected local documents (databases/computers), quick mathematical queries, aur live data fetch krne ka kaam kr rha hoon. Ye zyada complex query kisi aur se krwao aur bande k bachay ban jaao!

GENERAL/CONVERSATIONAL QUERY HANDLING:
If the query is simple, a greeting, or answerable directly from your internal knowledge without math, search, or retrieval, skip Action entirely and go straight to:
Thought: I can answer this directly.
Final Answer: [your direct answer]

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

def build_agent(return_intermediate_steps: bool = False):
    logger.info("Constructing high-speed Groq ReAct execution engine framework...")
    
    # Utilizing llama3-70b over Groq for bulletproof instruction following and low latency
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        groq_api_key=settings.GROQ_API_KEY, 
        temperature=0.1
    )
    
    tools = build_tools()
    prompt = PromptTemplate.from_template(REACT_PROMPT)

    agent = create_react_agent(llm, tools, prompt)
    
    # Max iterations capped at 4 to maximize latency efficiency and prevent loops
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=4,
        return_intermediate_steps=return_intermediate_steps,
    )
    return executor


def ask_agent(query: str) -> str:
    logger.info(f"Agent router receiving query event: {query}")
    try:
        executor = build_agent()
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