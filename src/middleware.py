# telemetry.py
import logging
import logfire

from src.config import Config


def init_observability(service_name: str = "rag-langgraph-app") -> None:
    """Centralized initializer for Pydantic Logfire and framework integrations."""
    
    # 1. Base Configuration
    logfire.configure(
        service_name=service_name,
        environment=Config.LOGFIRE_ENVIRONMENT,
    )

    # 2. Instrument integrations
    logfire.instrument_pydantic(record="all")  # Pydantic validation tracing goes here
    logfire.instrument_httpx()

    logfire.info("Logfire observability initialized successfully.")