from pathlib import Path
from typing import List, Union
from bs4 import BeautifulSoup
import logfire
from langchain_core.documents import Document


def load_html(
    file_path: Union[str, Path], 
    encoding: str = "utf-8"
) -> List[Document]:
    """Loads HTML files into LangChain Document objects by stripping DOM noise,

    handling multiple character encodings, and logging trace spans to Logfire.
    """
    path = Path(file_path).resolve()

    with logfire.span("load_html_file", file_path=str(path)) as span:
        
        # 1. Path Validations
        if not path.exists():
            logfire.error("HTML file does not exist", path=str(path))
            raise FileNotFoundError(f"HTML file not found at path: {path}")

        if not path.is_file():
            logfire.error("Provided path is a directory, not a file", path=str(path))
            raise IsADirectoryError(f"Expected a file path, but found directory: {path}")

        # 2. File Reading & Encoding Fallover Strategy
        raw_html = None
        encodings_to_try = [encoding, "utf-8", "utf-8-sig", "latin-1", "cp1252"]

        for enc in encodings_to_try:
            try:
                logfire.info("Reading HTML file", path=str(path), encoding=enc)
                with open(path, "r", encoding=enc) as f:
                    raw_html = f.read()
                break
            except UnicodeDecodeError:
                logfire.warning(
                    f"Encoding '{enc}' failed for HTML file, retrying next encoding",
                    path=str(path),
                )
                continue
            except PermissionError as e:
                logfire.error("Permission denied reading HTML file", path=str(path), error=str(e))
                raise PermissionError(f"Access denied for HTML file: {path}") from e

        if raw_html is None:
            logfire.error("Failed to decode HTML file with any supported encoding", path=str(path))
            raise UnicodeDecodeError(
                "unknown", b"", 0, 0, f"Could not decode HTML file {path.name} with standard encodings."
            )

        # 3. BeautifulSoup DOM Cleaning
        try:
            soup = BeautifulSoup(raw_html, "lxml")
        except Exception as parse_err:
            logfire.warning(
                "LXML parser failed, retrying with html.parser",
                path=str(path),
                error=str(parse_err),
            )
            try:
                soup = BeautifulSoup(raw_html, "html.parser")
            except Exception as fallback_err:
                logfire.error("Fatal HTML parsing error", path=str(path), error=str(fallback_err))
                raise RuntimeError(f"Could not parse HTML document ({path.name}): {fallback_err}") from fallback_err

        # Strip non-content / noisy tags
        for element in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "iframe"]):
            element.decompose()

        page_title = soup.title.string.strip() if soup.title and soup.title.string else ""

        # Extract normalized plain text
        lines = (line.strip() for line in soup.get_text(separator="\n").splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = "\n".join(chunk for chunk in chunks if chunk)

        # 4. Create Document Object
        metadata = {
            "file_name": path.name,
            "file_type": "html",
            "title": page_title,
            "char_length": len(clean_text),
        }

        logfire.info(
            "HTML loaded successfully",
            file_name=path.name,
            title=page_title,
            char_length=len(clean_text),
        )

        span.set_attribute("char_length", len(clean_text))
        span.set_attribute("page_title", page_title)

        return [Document(page_content=clean_text, metadata=metadata)]



def safe_load_html(file_path: str, encoding: str = "utf-8") -> List[Document]:
    """Production wrapper that safely logs errors with Logfire and returns an empty list on failure."""
    try:
        return load_html(file_path=file_path, encoding=encoding)
    except Exception as e:
        logfire.warning(
            "Skipping HTML document due to loading error", 
            file_path=file_path, 
            error_message=str(e)
        )
        return []