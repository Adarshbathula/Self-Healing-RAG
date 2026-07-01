from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.prompts import answer_prompt

logger = get_logger(__name__)


@lru_cache
def _get_answer_llm() -> ChatGroq:
    settings = get_settings()
    return ChatGroq(
        model=settings.model_name,
        api_key=settings.groq_api_key,
        temperature=0.2,
        max_tokens=1024,
    )


def generate_answer(question: str, context: str) -> str:
    llm = _get_answer_llm()
    chain = answer_prompt | llm | StrOutputParser()

    if not context or not context.strip():
        context = "(empty - no relevant documents were retrieved)"

    answer = chain.invoke({"context": context, "question": question}).strip()
    logger.info("Generated answer of length %s", len(answer))
    return answer
