from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from app.core.logging import get_logger

logger = get_logger(__name__)


def load_pdf(file_path: Path) -> list[Document]:
    logger.info("Loading PDF: %s", file_path)
    loader = PyPDFLoader(str(file_path))
    documents = loader.load()

    for doc in documents:
        doc.metadata["source"] = file_path.name

    logger.info("Loaded %s pages from %s", len(documents), file_path.name)
    return documents
