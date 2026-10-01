import uuid
from typing import List, Optional
import logfire
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    custom_separators: Optional[List[str]] = None,
) -> List[Document]:
    """Splits a list of LangChain Document objects into smaller overlapping chunks.

    Injects chunk metadata and captures tracing in Logfire.
    """
    if not documents:
        logfire.warning("Chunker received an empty list of documents.")
        return []

    # Default hierarchical separators: Paragraphs -> Headings -> Lines -> Words -> Characters
    separators = custom_separators or [
        "\n\n",
        "\n",
        "## ",
        "### ",
        ". ",
        " ",
        "",
    ]

    with logfire.span(
        "chunk_documents_execution",
        input_doc_count=len(documents),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    ) as span:

        logfire.info(
            "Initializing text splitter",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False,
        )

        try:
            # Perform text splitting
            chunked_docs = splitter.split_documents(documents)
        except Exception as e:
            logfire.exception("Error encountered during document chunking", error=str(e))
            raise RuntimeError(f"Failed to chunk documents: {e}") from e

        # Post-process chunks to inject unique IDs and chunk metadata
        for idx, chunk in enumerate(chunked_docs):
            source = chunk.metadata.get("source", "unknown")
            file_name = chunk.metadata.get("file_name", "unknown")

            # Enrich chunk metadata
            chunk.metadata["chunk_id"] = f"{file_name}_{idx}_{uuid.uuid4().hex[:8]}"
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["chunk_char_length"] = len(chunk.page_content)

        logfire.info(
            "Document chunking completed successfully",
            input_docs=len(documents),
            total_chunks=len(chunked_docs),
            avg_chunk_size=(
                sum(len(c.page_content) for c in chunked_docs) // len(chunked_docs)
                if chunked_docs
                else 0
            ),
        )

        span.set_attribute("total_chunks_created", len(chunked_docs))
        return chunked_docs


def safe_chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Production wrapper that safely catches errors and logs them to Logfire without crashing."""
    try:
        return chunk_documents(
            documents=documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except Exception as e:
        logfire.error("Safe chunking failed", error_message=str(e))
        return []