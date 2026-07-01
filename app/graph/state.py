from typing import TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    original_question: str
    rewritten_question: str
    active_question: str
    retrieved_documents: list[Document]
    retrieved_context: str
    answer: str
    retry_count: int
    max_retries: int
    context_quality: str
    sources: list[dict]
