from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    GROQ_API_KEY: str = Field(..., env="GROQ_API_KEY")
    TAVILY_API_KEY: str = Field(..., env="TAVILY_API_KEY")
    GEMINI_API_KEY: str = Field(..., env="GEMINI_API_KEY")
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()