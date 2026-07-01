from functools import lru_cache
from typing import Literal

from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.prompts import grader_prompt

logger = get_logger(__name__)


class GradeResult(BaseModel):
    verdict: Literal["GOOD", "POOR"] = Field(description="Overall grade of the retrieved context")
    reason: str = Field(description="Short justification for the verdict")


@lru_cache
def _get_grader_llm() -> ChatGroq:
    settings = get_settings()
    return ChatGroq(model=settings.model_name, api_key=settings.groq_api_key, temperature=0)


def grade_context(question: str, context: str) -> GradeResult:
    if not context or not context.strip():
        logger.info("Context is empty - grading as POOR without calling the LLM")
        return GradeResult(verdict="POOR", reason="Retrieved context is empty")

    llm = _get_grader_llm()
    structured_llm = llm.with_structured_output(GradeResult)
    chain = grader_prompt | structured_llm

    try:
        result: GradeResult = chain.invoke({"question": question, "context": context})
    except Exception:
        logger.exception("Structured grading failed, falling back to parser-based grading")
        result = _grade_with_parser(question, context, llm)

    logger.info("Context graded as %s - %s", result.verdict, result.reason)
    return result


def _grade_with_parser(question: str, context: str, llm: ChatGroq) -> GradeResult:
    parser = PydanticOutputParser(pydantic_object=GradeResult)
    chain = grader_prompt | llm | parser
    return chain.invoke({"question": question, "context": context})
