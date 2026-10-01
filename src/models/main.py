import os
import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import Config
import time
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq



# for vectorstore ingestion, we need to handle potential 429 errors from the Gemini API. The following class wraps the GoogleGenerativeAIEmbeddings to automatically retry on such errors with exponential backoff.
class ResilientGeminiEmbeddings(GoogleGenerativeAIEmbeddings):
    """GoogleGenerativeAIEmbeddings wrapper that automatically catches 429 errors

    and applies exponential backoff during vectorstore ingestion.
    """

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        reraise=True,
    )
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Pacing pause to stay under Requests Per Minute (RPM) quota
        time.sleep(1.0)
        return super().embed_documents(texts)


def get_gemini_embedder() -> GoogleGenerativeAIEmbeddings:
    """Creates and returns a Gemini embedder instance."""
    logfire.info("Initializing Gemini embedder...",model="models/gemini-embedding-001")
    return GoogleGenerativeAIEmbeddings(
        model=Config.EMBEDDINGS_MODEL,
        api_key=Config.GEMINI_API_KEY
    )

def get_resilient_gemini_embedder() -> ResilientGeminiEmbeddings:
    """Creates and returns a resilient Gemini embedder instance with retry logic."""
    logfire.info("Initializing Resilient Gemini embedder...",model="models/gemini-embedding-001")
    return ResilientGeminiEmbeddings(
        model=Config.EMBEDDINGS_MODEL,
        api_key=Config.GEMINI_API_KEY
    )





# -----------------------------------------------------------------------------
# LLM & Retriever Setup
# -----------------------------------------------------------------------------
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=Config.GEMINI_API_KEY,
        temperature=0.2,
    )


# -----------------------------------------------------------------------------
# LLM & Router Setup (Groq)
# -----------------------------------------------------------------------------
def get_groq_llm(model_name: str = "openai/gpt-oss-120b") -> ChatGroq:
    """Initializes Groq LLM instance using Llama 3.3 70B."""
    api_key = Config.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")

    return ChatGroq(
        model=model_name,
        groq_api_key=api_key,
        temperature=0.2,
    )


