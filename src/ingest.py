import os
from pathlib import Path
from datetime import datetime
from pypdf import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

# Clean system structure imports
from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DOC_CATEGORIES = {
    "Types_of_computers.pdf": "computers",
    "Types_of_database.pdf": "databases",
}

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def extract_pdf(filepath: Path) -> list[str]:
    """Extracts text from a PDF file and splits into overlapping semantic chunks."""
    logger.info(f"Extracting text from document asset: {filepath}")
    try:
        reader = PdfReader(str(filepath))
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

        if not full_text.strip():
            logger.warning(f"No extractable text found in PDF: {filepath}")
            return []

        chunks = text_splitter.split_text(full_text)
        logger.info(f"Generated {len(chunks)} overlapping chunks from {filepath.name}")
        return chunks
    except Exception as e:
        logger.error(f"Failed to read/parse PDF path {filepath}: {str(e)}", exc_info=True)
        return []


def build_vector_store() -> bool:
    """Builds or rebuilds the Chroma vector store from local PDF documents."""
    logger.info("Initializing Vector Database ingestion execution...")

    if not settings.GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is empty or missing in configuration. Halting ingestion.")
        return False

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    Path(settings.CHROMA_DIR).mkdir(parents=True, exist_ok=True)

    try:
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=settings.GEMINI_API_KEY,
        )
        client = chromadb.PersistentClient(path=settings.CHROMA_DIR)

        # Purge existing collections to ensure fresh indexing
        for col_name in [settings.CHROMA_COLLECTION, "day12_collection"]:
            try:
                client.delete_collection(col_name)
                logger.info(f"Purged existing collection '{col_name}'.")
            except Exception:
                pass

        collection = client.create_collection(settings.CHROMA_COLLECTION)

        all_documents, all_metadata = [], []
        for filename, category in DOC_CATEGORIES.items():
            filepath = DATA_DIR / filename
            if not filepath.exists():
                logger.warning(f"Target ingestion storage file not found, skipping target: {filepath}")
                continue

            raw_chunks = extract_pdf(filepath)
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
            return False

        ids = [f"id_{i}" for i in range(len(all_documents))]

        logger.info(f"Generating vectors via Gemini API for {len(all_documents)} chunks...")
        embeddings = embedding_model.embed_documents(all_documents)

        collection.add(
            documents=all_documents,
            embeddings=embeddings,
            metadatas=all_metadata,
            ids=ids,
        )
        logger.info(f"Successfully embedded and indexed {len(all_documents)} chunks into '{settings.CHROMA_COLLECTION}' at {settings.CHROMA_DIR}")
        return True

    except Exception as master_err:
        logger.critical(f"Critical breakdown in vector store execution pipeline: {str(master_err)}", exc_info=True)
        return False


if __name__ == "__main__":
    build_vector_store()