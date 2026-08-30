import ast
import math
import operator
import chromadb
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
try:
    from langchain_tavily import TavilySearch as TavilyTool
except ImportError:
    from langchain_community.tools.tavily_search import TavilySearchResults as TavilyTool

# Clean system structure imports
from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

# Singletons for connection pooling and reduced latency
_embedding_model = None
_chroma_client = None
_groq_llm = None


def _get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=settings.GEMINI_API_KEY,
        )
    return _embedding_model


def _get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
    return _chroma_client


def _get_groq_llm() -> ChatGroq:
    global _groq_llm
    if _groq_llm is None:
        _groq_llm = ChatGroq(
            model=settings.GROQ_COMPRESSION_MODEL,
            groq_api_key=settings.GROQ_API_KEY,
            temperature=0,
        )
    return _groq_llm


# ---------------------------------------------------------
# Secure AST-based Math Evaluator (No Insecure eval())
# ---------------------------------------------------------
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
    "abs": abs,
    "round": round,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "factorial": math.factorial,
}

_SAFE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}


def _eval_ast_node(node):
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
    elif isinstance(node, ast.Name):
        if node.id in _SAFE_CONSTANTS:
            return _SAFE_CONSTANTS[node.id]
        raise ValueError(f"Undefined variable or constant: '{node.id}'")
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type in _SAFE_OPERATORS:
            operand = _eval_ast_node(node.operand)
            return _SAFE_OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type in _SAFE_OPERATORS:
            left = _eval_ast_node(node.left)
            right = _eval_ast_node(node.right)
            if op_type is ast.Pow:
                if abs(right) > 10000 or abs(left) > 1e10:
                    raise ValueError("Exponent or base too large (DoS protection)")
            return _SAFE_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCTIONS:
            args = [_eval_ast_node(arg) for arg in node.args]
            if node.func.id == "factorial":
                if len(args) == 1 and args[0] > 1000:
                    raise ValueError("Factorial argument too large (DoS protection)")
            return _SAFE_FUNCTIONS[node.func.id](*args)
        raise ValueError(f"Function call not permitted: {getattr(node.func, 'id', 'unknown')}")
    else:
        raise ValueError(f"Unsupported expression syntax: {type(node).__name__}")


def safe_calculator(expression: str) -> str:
    """Safely evaluates mathematical expressions using an AST whitelist."""
    logger.info(f"Calculator tool triggered with expression: {expression}")
    # Strip any potential wrapping code blocks or whitespace
    clean_expr = expression.strip().strip("`").strip()
    try:
        parsed = ast.parse(clean_expr, mode="eval")
        result = _eval_ast_node(parsed)
        return str(result)
    except Exception as e:
        logger.warning(f"Calculator evaluation failure for '{clean_expr}': {str(e)}")
        return f"Error evaluating expression: {e}"


def retrieve_and_compress(query: str, k: int = 4) -> str:
    """Retrieves document fragments from Chroma and extracts facts via Groq in a single pass."""
    logger.info(f"KnowledgeBaseRetriever triggered for query: {query}")
    client = _get_chroma_client()
    
    collection = None
    for name in [settings.CHROMA_COLLECTION, "day12_collection"]:
        try:
            collection = client.get_collection(name)
            break
        except Exception:
            continue

    if collection is None:
        logger.error(f"Chroma collection not found under {settings.CHROMA_COLLECTION} or fallback.")
        return "Knowledge base index not initialized. Please verify chroma_db folder or run ingestion."

    try:
        embedding_model = _get_embedding_model()
        query_embed = embedding_model.embed_query(query)
        results = collection.query(query_embeddings=[query_embed], n_results=k)
    except Exception as e:
        logger.error(f"Embedding or query retrieval failed: {str(e)}", exc_info=True)
        return f"Error retrieving knowledge base documents: {str(e)}"

    if not results or not results.get("documents") or not results["documents"][0]:
        return "No relevant information found in the knowledge base."

    # Batch compression in a single prompt to minimize round trips and cut latency
    docs = [d.strip() for d in results["documents"][0] if d and d.strip()]
    if not docs:
        return "No relevant information found in the knowledge base."

    combined_context = "\n\n---\n\n".join(
        f"[Document Excerpt {i+1}]:\n{doc_text}"
        for i, doc_text in enumerate(docs)
    )

    logger.info(f"Starting single-pass context compression for {len(docs)} excerpts...")
    llm = _get_groq_llm()
    prompt = (
        "Extract ONLY the exact sentences, definitions, and facts from the following document context "
        "that directly answer or pertain to the target question. Keep the response factual and concise.\n"
        "If the document context contains no relevant info, respond with: 'No relevant information found.'\n\n"
        f"Question: {query}\n\n"
        f"Document Context:\n{combined_context}\n\n"
        "Relevant facts:"
    )

    try:
        response = llm.invoke(prompt)
        extracted = response.content.strip()
        logger.info("Context compression completed successfully.")
        return extracted if extracted else "No relevant info found after compression."
    except Exception as e:
        logger.error(f"Compression failed with Groq LPU: {str(e)}", exc_info=True)
        # Fallback to returning raw excerpts if compression fails
        return combined_context


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

    web_search_tool = TavilyTool(
        max_results=3,
        name="WebSearch",
        description="Use for current events, today's date, real-time facts, weather, or anything outside the local knowledge base.",
        tavily_api_key=settings.TAVILY_API_KEY,
    )

    return [calculator_tool, retriever_tool, web_search_tool]