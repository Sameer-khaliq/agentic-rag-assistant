import os
from pathlib import Path
from datetime import datetime
from pypdf import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb

# Naye systems se imports
from src_code.config import settings
from src_code.logger import get_logger

logger = get_logger(__name__)

# Relative reference from root configuration
PERSIST_DIR = "chroma_db"

DOC_CATEGORIES = {
    "data/Types_of_computers.pdf": "computers",
    "data/Types_of_database.pdf": "databases",
}


def extract_pdf(filepath: str, chunk_size: int = 500) -> list[str]:
    logger.info(f"Extracting text from document asset: {filepath}")
    try:
        reader = PdfReader(filepath)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

        words = full_text.split()
        chunks, current_chunk, current_length = [], [], 0
        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1
            if current_length >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_length = [], 0
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks
    except Exception as e:
        logger.error(f"Failed to read/parse PDF path {filepath}: {str(e)}", exc_info=True)
        return []


def build_vector_store():
    logger.info("Initializing Vector Database ingestion execution...")
    
    # Secure API verification from configuration block before proceeding
    if not os.environ.get("GEMINI_API_KEY"):
        # Pydantic validates this, but extra runtime fallback guarantees zero embedding failures
        logger.error("GEMINI_API_KEY environment binding is completely empty. Halting ingestion.")
        return

    # Ensure local data landing zone directory path exists safely
    Path("data").mkdir(exist_ok=True)

    try:
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=settings.GEMINI_API_KEY
        )
        client = chromadb.PersistentClient(path=PERSIST_DIR)

        try:
            client.delete_collection("day12_collection")
            logger.info("Purged existing outdated day12_collection mapping context.")
        except Exception:
            pass

        collection = client.create_collection("day12_collection")

        all_documents, all_metadata = [], []
        for filepath, category in DOC_CATEGORIES.items():
            if not os.path.exists(filepath):
                logger.warning(f"Target ingestion storage file not found, skipping target: {filepath}")
                continue

            raw_chunks = extract_pdf(filepath)
            filename = Path(filepath).name
            ingest_date = datetime.now().strftime("%Y-%m-%d")

            for chunk in raw_chunks:
                if chunk.strip():
                    all_documents.append(chunk)
                    all_metadata.append({
                        "source": filename,
                        "category": category,
                        "date": ingest_date,
                    })

        if not all_documents:
            logger.warning("Vector pipeline aborted: Zero clean document string fragments extracted.")
            return

        ids = [f"id_{i}" for i in range(len(all_documents))]
        
        logger.info(f"Generating vectors via Gemini API for {len(all_documents)} chunks...")
        embeddings = embedding_model.embed_documents(all_documents)

        collection.add(
            documents=all_documents,
            embeddings=embeddings,
            metadatas=all_metadata,
            ids=ids,
        )
        logger.info(f"Successfully embedded and indexed {len(all_documents)} chunks into storage layer path: {PERSIST_DIR}")
        
    except Exception as master_err:
        logger.critical(f"Critical breakdown in vector store execution pipeline: {str(master_err)}", exc_info=True)


if __name__ == "__main__":
    build_vector_store()