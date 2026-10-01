import time
import json
from pathlib import Path
from typing import List
import logfire
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 1. Automatic Retry Decorator for Rate Limits
@retry(
    stop=stop_after_attempt(6),  # Retry up to 6 times
    wait=wait_exponential(multiplier=2, min=4, max=60),  # Wait 4s, 8s, 16s, 32s, 60s
    retry_error_callback=lambda retry_state: logfire.warning(
        f"Rate limit or API error hit. Pausing and retrying (Attempt {retry_state.attempt_number})..."
    )
)
def embed_batch_with_retry(embedder: GoogleGenerativeAIEmbeddings, texts: List[str]) -> List[List[float]]:
    """Embeds a batch of texts with automatic pause and retry on rate limits."""
    return embedder.embed_documents(texts)


def embed_chunks_resumable(
    chunks: List[Document],
    embedder: GoogleGenerativeAIEmbeddings,
    batch_size: int = 20,
    delay_between_batches: float = 1.0,
    checkpoint_file: str = "embedding_progress.json"
) -> List[dict]:
    """Embeds chunks using Gemini safely by batching, handling 429 rate limits,

    and saving progress to disk so it can resume anytime.
    """
    checkpoint_path = Path(checkpoint_file)
    completed_records = []
    processed_chunk_ids = set()

    # Load existing checkpoint progress if it exists
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                completed_records = json.load(f)
                processed_chunk_ids = {item["chunk_id"] for item in completed_records}
            logfire.info(
                "Found existing checkpoint file", 
                completed_count=len(processed_chunk_ids)
            )
        except Exception as e:
            logfire.warning("Failed to load checkpoint file, starting fresh", error=str(e))

    # Filter out chunks that were already processed
    remaining_chunks = [c for c in chunks if c.metadata.get("chunk_id") not in processed_chunk_ids]

    if not remaining_chunks:
        logfire.info("All chunks are already embedded.")
        return completed_records

    logfire.info(
        "Starting resumable embedding process",
        total_chunks=len(chunks),
        remaining_chunks=len(remaining_chunks),
        batch_size=batch_size
    )

    with logfire.span("resumable_embedding_pipeline") as span:
        # Process in batches
        for i in range(0, len(remaining_chunks), batch_size):
            batch = remaining_chunks[i : i + batch_size]
            batch_texts = [doc.page_content for doc in batch]

            logfire.info(
                f"Embedding batch {i // batch_size + 1}",
                batch_size=len(batch),
                progress=f"{i + len(batch)}/{len(remaining_chunks)}"
            )

            # 1. Embed batch with rate-limit retry support
            vectors = embed_batch_with_retry(embedder, batch_texts)

            # 2. Append newly processed items
            for doc, vector in zip(batch, vectors):
                record = {
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "embedding": vector
                }
                completed_records.append(record)

            # 3. Save checkpoint to disk after every batch
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(completed_records, f, indent=2)

            logfire.info("Checkpoint saved", total_embedded=len(completed_records))

            # 4. Pacing pause to stay under Requests Per Minute (RPM) quota
            time.sleep(delay_between_batches)

        span.set_attribute("total_chunks_embedded", len(completed_records))
        return completed_records




