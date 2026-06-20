from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from tools import build_tools

REACT_PROMPT = """Answer the following question as best you can. You have access to the following tools:

{tools}

Use the following format:

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


def build_agent(model_name: str = "gemini-2.5-flash", return_intermediate_steps: bool = False):
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    tools = build_tools()
    prompt = PromptTemplate.from_template(REACT_PROMPT)

    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,
        return_intermediate_steps=return_intermediate_steps,
    )
    return executor


def ask_agent(query: str) -> str:
    executor = build_agent()
    result = executor.invoke({"input": query})
    return result["output"]


if __name__ == "__main__":
    answer = ask_agent("What is the difference between a centralized and distributed database?")
    print(f"\nFINAL ANSWER: {answer}")