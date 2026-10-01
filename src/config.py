from pathlib import Path
from typing import Literal
from pydantic import Field, HttpUrl, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

# Get the base directory of the project
BASE_DIR = Path(__file__).resolve().parent
# print(f"Base directory: {BASE_DIR}")


class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "My Awesome App"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False

    GROQ_API_KEY:str
    GROQ_FALLBACK_API_KEY:str
    
    QDRANT_API_KEY:str
    QDRANT_CLUSTER_ENDPOINT:str
    QDRANT_URL:str
    QDRANT_COLLECTION_NAME:str
    
    GEMINI_API_KEY:str

    LOGFIRE_TOKEN:str
    LOGFIRE_SERVICE_NAME:str
    LOGFIRE_ENVIRONMENT:str

    EMBEDDINGS_MODEL:str

   
    # Pydantic Settings Configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"  # Ignores extra environment variables not defined in the class
    )


# Instantiate settings (runs validation immediately)
Config = Settings()