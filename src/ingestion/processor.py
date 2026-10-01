from pathlib import Path
from typing import Dict, List, Callable, Union
import logfire
from langchain_core.documents import Document

# Import individual file loaders developed previously
from src.ingestion.loaders.pdf import safe_load_pdf
from src.ingestion.loaders.text import safe_load_text
from src.ingestion.loaders.office import safe_load_word_doc
from src.ingestion.loaders.pptx import safe_load_pptx
from src.ingestion.loaders.html import safe_load_html
from src.config import Config
import os
from typing import Optional, Union, Dict, Any
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from src.ingestion.chuncking.chunker import safe_chunk_documents
from src.ingestion.vectorstore.qdrant import store_documents_in_qdrant,upload_preembedded_records_to_qdrant
from src.models.main import get_gemini_embedder,get_resilient_gemini_embedder
from src.ingestion.retrival.embedder import embed_chunks_resumable

# Mapping file extensions to their corresponding safe loading functions
LOADER_MAPPING: Dict[str, Callable[[str], List[Document]]] = {
    ".pdf": safe_load_pdf,
    ".txt": safe_load_text,
    ".md": safe_load_text,
    ".log": safe_load_text,
    ".docx": safe_load_word_doc,
    ".doc": safe_load_word_doc,
    ".pptx": safe_load_pptx,
    ".ppt": safe_load_pptx,
    ".html": safe_load_html,
    ".htm": safe_load_html,
}


def load_documents_from_directory(
    directory_path: Union[str, Path], 
    recursive: bool = True
) -> List[Document]:
    """Scans a directory and loads all supported file types into LangChain Documents.

    
    Handles directory validation, file routing, exception isolation, and Logfire tracing.
    """
    folder = Path(directory_path).resolve()

    with logfire.span("load_directory", directory_path=str(folder), recursive=recursive) as span:
        
        # 1. Directory Validation
        if not folder.exists():
            logfire.error("Target directory does not exist", directory_path=str(folder))
            raise FileNotFoundError(f"Directory not found: {folder}")

        if not folder.is_dir():
            logfire.error("Provided path is not a directory", directory_path=str(folder))
            raise NotADirectoryError(f"Expected a directory path, but got: {folder}")

        # 2. Collect files
        glob_pattern = "**/*" if recursive else "*"
        all_files = [f for f in folder.glob(glob_pattern) if f.is_file()]

        if not all_files:
            logfire.warning("No files found in target directory", directory_path=str(folder))
            return []

        logfire.info(
            "Found files to scan", 
            total_files=len(all_files), 
            directory_path=str(folder)
        )

        all_documents: List[Document] = []
        processed_file_count = 0
        skipped_file_count = 0

        # 3. Iterate and route files
        for file_path in all_files:
            ext = file_path.suffix.lower()

            if ext not in LOADER_MAPPING:
                logfire.debug("Skipping unsupported file format", file_path=str(file_path), extension=ext)
                skipped_file_count += 1
                continue

            loader_func = LOADER_MAPPING[ext]
            
            # Load file safely (internal exceptions captured and logged to Logfire)
            docs = loader_func(str(file_path))

            if docs:
                # Add folder metadata
                for doc in docs:
                    doc.metadata["parent_directory"] = folder.name
                
                all_documents.extend(docs)
                processed_file_count += 1
            else:
                skipped_file_count += 1

        # 4. Final Telemetry Logging
        logfire.info(
            "Directory ingestion completed",
            directory_path=str(folder),
            processed_files=processed_file_count,
            skipped_files=skipped_file_count,
            total_chunks_created=len(all_documents),
        )

        span.set_attribute("processed_files", processed_file_count)
        span.set_attribute("skipped_files", skipped_file_count)
        span.set_attribute("total_documents", len(all_documents))

        return all_documents




def run_ingestion_pipeline(
    data_directory: Union[str, Path],
    collection_name: str = "rag_documents",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    batch_size: int = 20,
    recursive: bool = True,
    qdrant_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs the full document ingestion pipeline end-to-end using pre-embedded vectors."""
    input_path = Path(data_directory).resolve()

    with logfire.span(
        "full_ingestion_pipeline",
        data_directory=str(input_path),
        collection_name=collection_name,
    ) as span:

        logfire.info("Starting document ingestion pipeline", directory=str(input_path))

        # -------------------------------------------------------------------------
        # Step 1: Scan Directory & Load Documents
        # -------------------------------------------------------------------------
        raw_docs = load_documents_from_directory(
            directory_path=input_path, recursive=recursive
        )

        if not raw_docs:
            logfire.warning("Ingestion aborted: No valid documents found.")
            return {"status": "warning", "message": "No valid documents loaded.", "chunks_indexed": 0}

        # -------------------------------------------------------------------------
        # Step 2: Chunk Documents
        # -------------------------------------------------------------------------
        chunks = safe_chunk_documents(
            documents=raw_docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if not chunks:
            logfire.error("Ingestion failed: Zero text chunks produced.")
            return {"status": "error", "message": "Chunking failed.", "chunks_indexed": 0}

        # -------------------------------------------------------------------------
        # Step 3: Embed Chunks Safely (Handles Gemini 429 Rate-Limits & Checkpoints)
        # -------------------------------------------------------------------------
        api_key = Config.GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing.")

        embedder = get_gemini_embedder()

        logfire.info("Embedding chunks with rate-limit handling & state checkpointing...")
        
        # Returns list of dicts with pre-computed vectors
        preembedded_records = embed_chunks_resumable(
            chunks=chunks,
            embedder=embedder,
            batch_size=batch_size,
            delay_between_batches=1.0,
            checkpoint_file="embedding_progress.json"
        )

        # -------------------------------------------------------------------------
        # Step 4: Upload Pre-Embedded Vectors Directly to Qdrant (No API Calls!)
        # -------------------------------------------------------------------------
        qdrant_url = Config.QDRANT_CLUSTER_ENDPOINT
        qdrant_api_key = Config.QDRANT_API_KEY

        logfire.info("Uploading vectors directly to Qdrant...", total_records=len(preembedded_records))
        
        upload_preembedded_records_to_qdrant(
            records=preembedded_records,
            collection_name=collection_name,
            qdrant_url=Config.QDRANT_CLUSTER_ENDPOINT,
            qdrant_api_key=Config.QDRANT_API_KEY,
            qdrant_path=qdrant_path,  # Fallback local path if URL is None
            batch_size=100
        )

        # -------------------------------------------------------------------------
        # Step 5: Summary Metrics
        # -------------------------------------------------------------------------
        metrics = {
            "status": "success",
            "collection_name": collection_name,
            "raw_documents_loaded": len(raw_docs),
            "chunks_indexed": len(preembedded_records),
            "qdrant_target": qdrant_url or qdrant_path or "./qdrant_db",
        }

        logfire.info("Ingestion pipeline completed successfully!", **metrics)
        span.set_attribute("chunks_indexed", len(preembedded_records))

        return metrics