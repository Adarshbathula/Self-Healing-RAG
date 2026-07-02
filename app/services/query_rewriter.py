from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.prompts import rewriter_prompt

logger = get_logger(__name__)


@lru_cache
def _get_rewriter_llm() -> ChatGroq:
    settings = get_settings()
    return ChatGroq(model=settings.fast_model_name, api_key=settings.groq_api_key, temperature=0.3)


def rewrite_query(original_question: str, active_question: str) -> str:
    llm = _get_rewriter_llm()
    chain = rewriter_prompt | llm | StrOutputParser()

    rewritten = chain.invoke(
        {"original_question": original_question, "active_question": active_question}
    ).strip()

    logger.info("Rewrote query %r -> %r", active_question, rewritten)
    return rewritten
