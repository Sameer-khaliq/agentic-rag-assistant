from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ensure environment variables from .env are mirrored into os.environ
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    GROQ_API_KEY: str
    TAVILY_API_KEY: str
    GEMINI_API_KEY: str
    LOG_LEVEL: str = "INFO"
    CHROMA_DIR: str = str(BASE_DIR / "chroma_db")
    CHROMA_COLLECTION: str = "agentic_rag_docs"
    GROQ_AGENT_MODEL: str = "openai/gpt-oss-120b"
    GROQ_COMPRESSION_MODEL: str = "openai/gpt-oss-20b"

settings = Settings()