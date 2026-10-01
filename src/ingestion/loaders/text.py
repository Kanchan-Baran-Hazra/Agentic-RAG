from pathlib import Path
from typing import List, Union
import logfire
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader


def load_text(
    file_path: Union[str, Path], 
    encoding: str = "utf-8"
) -> List[Document]:
    """Loads a plain text or markdown file into LangChain Document objects with

    encoding fallback, exception handling, and Logfire tracing.
    """
    path = Path(file_path).resolve()

    with logfire.span("load_text_file", file_path=str(path)) as span:
        
        # 1. Path validations
        if not path.exists():
            logfire.error("Text file does not exist", path=str(path))
            raise FileNotFoundError(f"Text file not found at path: {path}")

        if not path.is_file():
            logfire.error("Provided path is a directory, not a file", path=str(path))
            raise IsADirectoryError(f"Expected a file path, but found directory: {path}")

        # 2. File Reading & Encoding Fallover Strategy
        encodings_to_try = [encoding, "utf-8", "utf-8-sig", "latin-1", "cp1252"]
        documents = None

        for enc in encodings_to_try:
            try:
                logfire.info("Attempting to parse text file", path=str(path), encoding=enc)
                loader = TextLoader(file_path=str(path), encoding=enc)
                documents = loader.load()
                break  # Successfully loaded, exit loop
            except UnicodeDecodeError:
                logfire.warning(f"Encoding '{enc}' failed for text file, retrying next encoding", path=str(path))
                continue

        if documents is None:
            logfire.error("Failed to decode text file with any supported encoding", path=str(path))
            raise UnicodeDecodeError(
                "unknown", b"", 0, 0, f"Could not decode file {path.name} with standard encodings."
            )

        # 3. Enrich metadata
        for doc in documents:
            doc.metadata["file_name"] = path.name
            doc.metadata["file_type"] = path.suffix.strip(".").lower() or "txt"

        logfire.info(
            "Text file loaded successfully",
            file_name=path.name,
            char_count=len(documents[0].page_content) if documents else 0
        )

        span.set_attribute("char_length", len(documents[0].page_content) if documents else 0)
        return documents



def safe_load_text(file_path: str, encoding: str = "utf-8") -> List[Document]:
    """Production wrapper that safely logs errors with Logfire and returns an empty list on failure."""
    try:
        return load_text(file_path=file_path, encoding=encoding)
    except Exception as e:
        logfire.warning(
            "Skipping text file due to loading error", 
            file_path=file_path, 
            error_message=str(e)
        )
        return []