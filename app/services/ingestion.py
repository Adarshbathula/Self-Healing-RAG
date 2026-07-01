from pathlib import Path

from fastapi import UploadFile

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.loader import load_pdf
from app.services.splitter import split_documents
from app.services.vector_store import get_vector_store_manager

logger = get_logger(__name__)


async def save_upload(upload: UploadFile) -> Path:
    settings = get_settings()
    destination = settings.documents_path / upload.filename

    content = await upload.read()
    destination.write_bytes(content)
    logger.info("Saved uploaded file to %s", destination)
    return destination


async def ingest_uploaded_file(upload: UploadFile) -> int:
    file_path = await save_upload(upload)
    documents = load_pdf(file_path)
    chunks = split_documents(documents)

    manager = get_vector_store_manager()
    added = manager.add_documents(chunks)
    return added
