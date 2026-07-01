import threading

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.embeddings import get_embeddings

logger = get_logger(__name__)


class VectorStoreManager:
    """Thread-safe singleton wrapper around a FAISS vector store with disk persistence."""

    _instance: "VectorStoreManager | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._settings = get_settings()
        self._embeddings = get_embeddings()
        self._store: FAISS | None = None
        self._store_lock = threading.Lock()
        self._load_from_disk()

    @classmethod
    def get_instance(cls) -> "VectorStoreManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _index_files_exist(self) -> bool:
        index_dir = self._settings.faiss_index_dir
        return (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists()

    def _load_from_disk(self) -> None:
        if not self._index_files_exist():
            logger.info("No existing FAISS index found at %s", self._settings.faiss_index_dir)
            return

        try:
            self._store = FAISS.load_local(
                str(self._settings.faiss_index_dir),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("Loaded existing FAISS index from %s", self._settings.faiss_index_dir)
        except Exception:
            logger.exception("Failed to load FAISS index from disk, starting fresh")
            self._store = None

    def add_documents(self, documents: list[Document]) -> int:
        if not documents:
            return 0

        with self._store_lock:
            if self._store is None:
                self._store = FAISS.from_documents(documents, self._embeddings)
            else:
                self._store.add_documents(documents)
            self._store.save_local(str(self._settings.faiss_index_dir))

        logger.info("Added %s chunks to FAISS index and persisted to disk", len(documents))
        return len(documents)

    def similarity_search(self, query: str, k: int | None = None) -> list[Document]:
        if self._store is None:
            logger.warning("Similarity search requested but vector store is empty")
            return []

        top_k = k or self._settings.top_k
        return self._store.similarity_search(query, k=top_k)

    @property
    def is_ready(self) -> bool:
        return self._store is not None

    @property
    def document_count(self) -> int:
        if self._store is None:
            return 0
        return self._store.index.ntotal


def get_vector_store_manager() -> VectorStoreManager:
    return VectorStoreManager.get_instance()
