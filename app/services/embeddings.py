from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    settings = get_settings()
    logger.info("Loading embedding model: %s", settings.embedding_model)
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )
