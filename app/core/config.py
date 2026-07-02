from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    groq_api_key: str
    model_name: str = "llama-3.3-70b-versatile"
    fast_model_name: str = "llama-3.1-8b-instant"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    faiss_index_path: str = "faiss_index"
    documents_dir: str = "documents"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4
    max_retries: int = 3
    log_level: str = "INFO"

    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440
    users_db_path: str = "users.db"

    @property
    def faiss_index_dir(self) -> Path:
        path = Path(self.faiss_index_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def documents_path(self) -> Path:
        path = Path(self.documents_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()
