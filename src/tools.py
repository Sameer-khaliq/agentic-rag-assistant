import math
import chromadb
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()


CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "day12_collection"


def safe_calculator(expression: str) -> str:
   
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


def retrieve_and_compress(query: str, k: int = 4) -> str:
    
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        return "Knowledge base index not initialized. Please verify chroma_db folder."
        
    query_embed = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embed], n_results=k)
    
    if not results or not results.get("documents") or not results["documents"][0]:
        return "No relevant information found in the knowledge base."

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    compressed_parts = []
    
    for doc_text in results["documents"][0]:
        prompt = (
            f"Extract ONLY the exact sentences, facts, or parts from the context "
            f"that are directly relevant to answering the target question.\n"
            f"If the document context contains no relevant info, respond with exactly an empty string.\n\n"
            f"Question: {query}\n"
            f"Document Context:\n{doc_text}\n\n"
            f"Relevant facts:"
        )
        response = llm.invoke(prompt)
        extracted = response.content.strip()
        if extracted and extracted != "''":
            compressed_parts.append(extracted)

    return "\n\n".join(compressed_parts) if compressed_parts else "No relevant info found after compression."


def build_tools() -> list[Tool]:
    calculator_tool = Tool(
        name="Calculator",
        func=safe_calculator,
        description="Use for any math calculation. Input must be a valid Python math expression, e.g. '847 * 23' or 'sqrt(1764)'.",
    )

    retriever_tool = Tool(
        name="KnowledgeBaseRetriever",
        func=retrieve_and_compress,
        description=(
            "Use for questions about types of computers or types of databases — "
            "this is the only source of that information. Input should be the clear question."
        ),
    )

    web_search_tool = TavilySearchResults(
        max_results=3,
        name="WebSearch",
        description="Use for current events, today's date, real-time facts, weather, or anything outside the local knowledge base.",
    )

    return [calculator_tool, retriever_tool, web_search_tool]