from langchain_core.documents import Document

from app.core.logging import get_logger
from app.services.vector_store import get_vector_store_manager

logger = get_logger(__name__)


def retrieve_documents(query: str, k: int | None = None) -> list[Document]:
    manager = get_vector_store_manager()
    documents = manager.similarity_search(query, k=k)
    logger.info("Retrieved %s documents for query: %s", len(documents), query)
    return documents


def format_context(documents: list[Document]) -> str:
    if not documents:
        return ""

    seen: set[str] = set()
    blocks: list[str] = []
    for doc in documents:
        normalized = " ".join(doc.page_content.split())
        if normalized in seen:
            continue
        seen.add(normalized)

        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(f"[Source: {source} | Page: {page}]\n{doc.page_content}")

    return "\n\n---\n\n".join(blocks)


def build_sources(documents: list[Document]) -> list[dict]:
    sources: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", -1)
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"page": page, "source": source, "content": doc.page_content})
    return sources
