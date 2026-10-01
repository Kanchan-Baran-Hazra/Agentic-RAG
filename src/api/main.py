import os
import logfire
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agents.graph import build_rag_graph
from src.agents.state import create_initial_state
from src.models.main import get_gemini_embedder
from src import main

# Load environment variables
load_dotenv()

# Global state holder for compiled graph
app_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events for loading models/graph."""
    logfire.info("Initializing Agentic RAG components...")

    # Initialize Embeddings & Build LangGraph Runnable
    embeddings = get_gemini_embedder()
    app_state["agent"] = build_rag_graph(embeddings=embeddings)
    logfire.info("Agentic RAG pipeline successfully compiled!")

    yield

    # Clean up resources on shutdown
    app_state.clear()
    logfire.info("Shutting down Agentic RAG API.")


# -----------------------------------------------------------------------------
# FastAPI App Setup
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Agentic RAG API",
    description="LangGraph + Groq + Qdrant Agentic RAG Pipeline API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------
class DocumentMetadata(BaseModel):
    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        description="User query string",
        example="What are the key technical concepts in the documents?",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metadata context or request tags"
    )


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_documents: List[DocumentMetadata]
    document_count: int


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint to verify service readiness."""
    return {"status": "ok", "agent_loaded": "agent" in app_state}


@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
)
async def run_query(request: QueryRequest):
    """Executes the Agentic RAG Graph pipeline for an incoming user question."""
    agent = app_state.get("agent")
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent pipeline is not initialized.",
        )

    try:
        # Create initial state dictionary
        initial_state = create_initial_state(
            question=request.question,
            metadata=request.metadata,
        )

        # Invoke the LangGraph pipeline
        result = await agent.ainvoke(initial_state)

        # Extract output documents and formatting
        docs = result.get("documents", [])
        formatted_docs = [
            DocumentMetadata(
                page_content=doc.page_content,
                metadata=doc.metadata,
            )
            for doc in docs
        ]

        return QueryResponse(
            question=request.question,
            answer=result.get("generation", "No response generated."),
            retrieved_documents=formatted_docs,
            document_count=len(formatted_docs),
        )

    except Exception as e:
        logfire.error("Error processing RAG query", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution error: {str(e)}",
        )