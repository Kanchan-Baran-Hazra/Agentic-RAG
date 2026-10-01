from pathlib import Path
from typing import List, Union
import logfire
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredPowerPointLoader


def load_pptx(file_path: Union[str, Path]) -> List[Document]:
    """Loads PowerPoint presentations (.pptx or .ppt) into LangChain Document objects

    with slide-by-slide metadata extraction, exception handling, and Logfire tracing.
    """
    path = Path(file_path).resolve()

    with logfire.span("load_pptx_file", file_path=str(path)) as span:
        
        # 1. Path Validations
        if not path.exists():
            logfire.error("PowerPoint file does not exist", path=str(path))
            raise FileNotFoundError(f"PowerPoint file not found at path: {path}")

        if not path.is_file():
            logfire.error("Provided path is a directory, not a file", path=str(path))
            raise IsADirectoryError(f"Expected a file path, but found directory: {path}")

        suffix = path.suffix.lower()
        if suffix not in [".pptx", ".ppt"]:
            logfire.error("Unsupported PowerPoint extension", suffix=suffix, path=str(path))
            raise ValueError(f"Extension '{suffix}' is not supported as a PowerPoint presentation.")

        documents: List[Document] = []

        # 2. Strategy: Direct python-pptx extraction for .pptx files (Fast & lightweight)
        if suffix == ".pptx":
            try:
                logfire.info("Parsing PPTX with python-pptx", path=str(path))
                from pptx import Presentation

                prs = Presentation(str(path))
                for slide_idx, slide in enumerate(prs.slides, start=1):
                    text_runs = []
                    for shape in slide.shapes:
                        if hasattr(shape, "text") and shape.text:
                            text_runs.append(shape.text.strip())

                    slide_content = "\n".join(text_runs).strip()
                    if slide_content:
                        documents.append(
                            Document(
                                page_content=slide_content,
                                metadata={
                                    "file_name": path.name,
                                    "file_type": "pptx",
                                    "slide_number": slide_idx,
                                },
                            )
                        )
            except Exception as e:
                logfire.warning(
                    "python-pptx parsing failed, falling back to UnstructuredPowerPointLoader",
                    path=str(path),
                    error=str(e),
                )

        # 3. Fallback or primary loader for legacy .ppt / complex elements
        if not documents:
            try:
                logfire.info("Attempting load with UnstructuredPowerPointLoader", path=str(path))
                loader = UnstructuredPowerPointLoader(file_path=str(path), mode="single")
                documents = loader.load()

                for doc in documents:
                    doc.metadata["file_name"] = path.name
                    doc.metadata["file_type"] = suffix.strip(".")
            except Exception as e:
                logfire.error("UnstructuredPowerPointLoader failed", path=str(path), error=str(e))
                raise RuntimeError(f"Failed to extract content from PowerPoint ({path.name}): {e}") from e

        total_chars = sum(len(doc.page_content) for doc in documents)

        logfire.info(
            "PowerPoint loaded successfully",
            file_name=path.name,
            slide_count=len(documents),
            total_chars=total_chars,
        )

        span.set_attribute("slide_count", len(documents))
        span.set_attribute("total_chars", total_chars)

        return documents


def safe_load_pptx(file_path: str) -> List[Document]:
    """Production wrapper that safely logs errors with Logfire and returns an empty list on failure."""
    try:
        return load_pptx(file_path=file_path)
    except Exception as e:
        logfire.warning(
            "Skipping PowerPoint presentation due to loading error", 
            file_path=file_path, 
            error_message=str(e)
        )
        return []