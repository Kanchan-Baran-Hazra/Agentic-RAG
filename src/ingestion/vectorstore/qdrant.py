import os
from typing import Dict, List, Optional, Union,Any, Callable
from pathlib import Path
import logfire
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from src.config import Config
from qdrant_client.http import models
import uuid


def get_qdrant_client(
    path: Optional[str] = "./qdrant_db",
    url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> QdrantClient:
    """Initializes a Qdrant Client for local storage or remote cloud deployment."""
    # 1. Use Remote / Cloud endpoint if provided
    if url or os.getenv("QDRANT_URL"):
        q_url = url or os.getenv("QDRANT_URL")
        q_key = api_key or os.getenv("QDRANT_API_KEY")
        logfire.info("Connecting to Qdrant Cloud / Remote Server", url=q_url)
        return QdrantClient(url=q_url, api_key=q_key)

    # 2. Local Disk Storage (Default)
    logfire.info("Using local Qdrant disk storage", path=path)
    return QdrantClient(path=path)


def store_documents_in_qdrant(
    documents: List[Document],
    embeddings: Embeddings,
    collection_name: str = "rag_documents",
    qdrant_path: Optional[str] = None,
    batch_size: int = 50,
    q_url: Optional[str] = None,
    q_key: Optional[str] = None,
) -> QdrantVectorStore:
    """Indexes document chunks into Qdrant using either Qdrant Cloud or Local path."""
    if not documents:
        raise ValueError("Document list cannot be empty.")

    with logfire.span("qdrant_indexing", total_documents=len(documents)):
        
        # Check environment variables for Qdrant Cloud
        # q_url = os.getenv("QDRANT_URL")
        # q_key = os.getenv("QDRANT_API_KEY")

        if q_url:
            logfire.info("Indexing directly into Qdrant Cloud", url=q_url)
            vector_store = QdrantVectorStore.from_documents(
                documents=documents,
                embedding=embeddings,
                url=q_url,
                api_key=q_key,
                collection_name=collection_name,
                batch_size=batch_size,
            )
        else:
            local_path = qdrant_path or "./qdrant_db"
            logfire.info("Indexing into Local Qdrant Disk Storage", path=local_path)
            vector_store = QdrantVectorStore.from_documents(
                documents=documents,
                embedding=embeddings,
                path=local_path,
                collection_name=collection_name,
                batch_size=batch_size,
            )

        logfire.info("Documents successfully indexed in Qdrant", collection=collection_name)
        return vector_store


def load_existing_qdrant_store(
    embeddings: Embeddings,
    collection_name: str = "rag_documents",
    qdrant_path: Optional[str] = None,
) -> QdrantVectorStore:
    """Loads an existing Qdrant collection for query retrieval."""
    with logfire.span("load_qdrant_store", collection=collection_name):
        
        # 1. Use Qdrant Cloud if endpoint is configured
        if Config.QDRANT_CLUSTER_ENDPOINT:
            return QdrantVectorStore.from_existing_collection(
                embedding=embeddings,
                collection_name=collection_name,
                url=Config.QDRANT_CLUSTER_ENDPOINT,
                api_key=Config.QDRANT_API_KEY,
            )
        
        # 2. Fall back to Local Disk storage
        local_path = qdrant_path or "./qdrant_db"
        return QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=collection_name,
            path=local_path,
        )




def upload_preembedded_records_to_qdrant(
    records: List[Dict[str, Any]],
    collection_name: str = "rag_documents",
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    qdrant_path: Optional[str] = None,
    batch_size: int = 25,  # Reduced batch size to prevent timeouts
):
    """Uploads pre-embedded chunk records directly to Qdrant Cloud or local disk."""
    if not records:
        logfire.warning("No records provided to upload to Qdrant.")
        return

    url = qdrant_url or os.getenv("QDRANT_URL")
    api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")

    if url:
        logfire.info("Connecting to Qdrant Cloud / Remote Server", url=url)
        # 1. Increased timeout from default 5.0s to 60.0s
        client = QdrantClient(url=url, api_key=api_key, timeout=60.0)
    else:
        disk_path = qdrant_path or "./qdrant_db"
        logfire.info("Connecting to Local Qdrant Disk Storage", path=disk_path)
        client = QdrantClient(path=disk_path)

    # 2. Ensure Collection Exists
    if not client.collection_exists(collection_name):
        vector_size = len(records[0]["embedding"])
        logfire.info("Creating new Qdrant collection", collection=collection_name, size=vector_size)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )

    # 3. Format Records as Qdrant Points (UUID v5)
    points = []
    for rec in records:
        payload = {
            "page_content": rec["page_content"],
            **rec.get("metadata", {})
        }
        
        chunk_str_id = rec.get("chunk_id", "")
        payload["original_chunk_id"] = chunk_str_id

        if chunk_str_id:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_str_id))
        else:
            point_id = str(uuid.uuid4())

        points.append(
            models.PointStruct(
                id=point_id,
                vector=rec["embedding"],
                payload=payload
            )
        )

    # 4. Upload Points in Smaller Batches
    with logfire.span("qdrant_upload", total_points=len(points)):
        client.upload_points(
            collection_name=collection_name,
            points=points,
            batch_size=batch_size, # Smaller batch size prevents HTTP request timeouts
            wait=True
        )

    logfire.info("Direct Qdrant upload complete!", inserted=len(points))