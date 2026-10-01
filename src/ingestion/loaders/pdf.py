from pathlib import Path
from typing import List, Optional, Union
import logfire
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from pypdf.errors import PdfReadError, PyPdfError


def load_pdf(
    file_path: Union[str, Path], 
    password: Optional[str] = None
) -> List[Document]:
    """Loads and parses a PDF document into LangChain Document objects with

    full exception handling and Logfire tracing.
    """
    path = Path(file_path).resolve()

    # Wrap the entire operation in a Logfire span
    with logfire.span("load_pdf_file", file_path=str(path)) as span:
        
        # 1. File existence validation
        if not path.exists():
            logfire.error("PDF file does not exist", path=str(path))
            raise FileNotFoundError(f"PDF file not found at path: {path}")

        if not path.is_file():
            logfire.error("Path provided is not a file", path=str(path))
            raise IsADirectoryError(f"Expected a file path, but found directory: {path}")

        # 2. File parsing and extraction
        try:
            logfire.info("Parsing PDF file", path=str(path))
            loader = PyPDFLoader(file_path=str(path), password=password)
            documents = loader.load()

            # Enrich metadata
            for doc in documents:
                doc.metadata["file_name"] = path.name
                doc.metadata["file_type"] = "pdf"

            logfire.info(
                "PDF loaded successfully",
                file_name=path.name,
                page_count=len(documents)
            )
            
            span.set_attribute("pages_loaded", len(documents))
            return documents

        # 3. Exception Handling
        except PermissionError as e:
            logfire.error("Permission denied reading file", path=str(path), error=str(e))
            raise PermissionError(f"Access denied for PDF file: {path}") from e

        except PdfReadError as e:
            # Handles corrupt files, invalid PDF headers, or incorrect passwords
            logfire.error("Failed to read PDF format or wrong password", path=str(path), error=str(e))
            raise ValueError(f"Corrupt or encrypted PDF file ({path.name}): {e}") from e

        except PyPdfError as e:
            # General PyPDF extraction error
            logfire.error("PyPDF processing error", path=str(path), error=str(e))
            raise RuntimeError(f"Error extracting content from PDF ({path.name}): {e}") from e

        except Exception as e:
            # Fallback for unexpected failures
            logfire.exception("Unexpected error while loading PDF", path=str(path))
            raise RuntimeError(f"Unexpected error loading PDF ({path.name}): {e}") from e


def safe_load_pdf(file_path: str, password: Optional[str] = None) -> List[Document]:
    """Production wrapper that safely logs errors with Logfire and returns an empty list on failure."""
    try:
        return load_pdf(file_path=file_path, password=password)
    except Exception as e:
        logfire.warning(
            "Skipping PDF due to loading error", 
            file_path=file_path, 
            error_message=str(e)
        )
        return []