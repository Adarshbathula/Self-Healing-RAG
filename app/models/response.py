from pydantic import BaseModel


class SourceChunk(BaseModel):
    page: int
    source: str
    content: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    retries: int
    context_quality: str


class UploadedFile(BaseModel):
    filename: str
    chunks_added: int


class UploadResponse(BaseModel):
    message: str
    files: list[UploadedFile]
    total_chunks_added: int


class HealthResponse(BaseModel):
    status: str
    vector_store_ready: bool
    indexed_documents: int
