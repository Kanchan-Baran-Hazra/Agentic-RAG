from pathlib import Path
from typing import List, Union
import logfire
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredWordDocumentLoader


def load_word_doc(file_path: Union[str, Path]) -> List[Document]:
    """Loads Word documents (.docx or .doc) into LangChain Document objects with

    automatic loader fallback, exception handling, and Logfire tracing.
    """
    path = Path(file_path).resolve()

    with logfire.span("load_word_document", file_path=str(path)) as span:
        
        # 1. Path validations
        if not path.exists():
            logfire.error("Word file does not exist", path=str(path))
            raise FileNotFoundError(f"Word file not found at path: {path}")

        if not path.is_file():
            logfire.error("Provided path is a directory, not a file", path=str(path))
            raise IsADirectoryError(f"Expected a file path, but found directory: {path}")

        suffix = path.suffix.lower()
        if suffix not in [".docx", ".doc"]:
            logfire.error("Unsupported Word extension", suffix=suffix, path=str(path))
            raise ValueError(f"File extension '{suffix}' is not supported as a Word document.")

        documents: List[Document] = []

        # 2. Strategy: Try Docx2txtLoader first (faster, lightweight)
        if suffix == ".docx":
            try:
                logfire.info("Attempting load with Docx2txtLoader", path=str(path))
                loader = Docx2txtLoader(file_path=str(path))
                documents = loader.load()
            except Exception as e:
                logfire.warning(
                    "Docx2txtLoader failed, falling back to UnstructuredWordDocumentLoader",
                    path=str(path),
                    error=str(e),
                )

        # 3. Fallback or primary loader for legacy .doc files
        if not documents:
            try:
                logfire.info("Attempting load with UnstructuredWordDocumentLoader", path=str(path))
                loader = UnstructuredWordDocumentLoader(file_path=str(path), mode="single")
                documents = loader.load()
            except Exception as e:
                logfire.error("UnstructuredWordDocumentLoader failed", path=str(path), error=str(e))
                raise RuntimeError(f"Failed to extract content from Word document ({path.name}): {e}") from e

        # 4. Enrich metadata
        for doc in documents:
            doc.metadata["file_name"] = path.name
            doc.metadata["file_type"] = suffix.strip(".")

        char_count = sum(len(doc.page_content) for doc in documents)
        
        logfire.info(
            "Word document loaded successfully",
            file_name=path.name,
            doc_count=len(documents),
            total_chars=char_count,
        )

        span.set_attribute("doc_count", len(documents))
        span.set_attribute("total_chars", char_count)

        return documents


def safe_load_word_doc(file_path: str) -> List[Document]:
    """Production wrapper that safely logs errors with Logfire and returns an empty list on failure."""
    try:
        return load_word_doc(file_path=file_path)
    except Exception as e:
        logfire.warning(
            "Skipping Word document due to loading error", 
            file_path=file_path, 
            error_message=str(e)
        )
        return []