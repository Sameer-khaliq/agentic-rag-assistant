import math
import chromadb
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults

# Clean system structure imports
from src_code.config import settings
from src_code.logger import get_logger

logger = get_logger(__name__)

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "day12_collection"


def safe_calculator(expression: str) -> str:
    logger.info(f"Calculator tool triggered with execution payload: {expression}")
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        logger.error(f"Calculator computation failure: {str(e)}")
        return f"Error evaluating expression: {e}"


def retrieve_and_compress(query: str, k: int = 4) -> str:
    logger.info(f"KnowledgeBaseRetriever pipeline triggered for query: {query}")
    
    # Keeping Gemini for embeddings since the vector store expects it
    embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=settings.GEMINI_API_KEY
)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        logger.error(f"Failed to access Chroma collection context: {str(e)}")
        return "Knowledge base index not initialized. Please verify chroma_db folder."
        
    query_embed = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embed], n_results=k)
    
    if not results or not results.get("documents") or not results["documents"][0]:
        return "No relevant information found in the knowledge base."

    # Sub-second latency optimized compression step via Groq
    logger.info("Starting context compression map loop using Groq LPU...")
    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=settings.GROQ_API_KEY, temperature=0)
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

    logger.info(f"Context compression complete. Extracted {len(compressed_parts)} valid reference pieces.")
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
        tavily_api_key=settings.TAVILY_API_KEY
    )

    return [calculator_tool, retriever_tool, web_search_tool]