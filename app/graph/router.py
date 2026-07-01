from typing import Literal

from app.core.logging import get_logger
from app.graph.state import GraphState

logger = get_logger(__name__)


def route_after_grading(state: GraphState) -> Literal["generate", "rewrite"]:
    if state["context_quality"] == "GOOD":
        logger.info("Context graded GOOD - routing to generate")
        return "generate"

    if state["retry_count"] >= state["max_retries"]:
        logger.warning(
            "Context graded POOR but max retries (%s) reached - routing to generate anyway",
            state["max_retries"],
        )
        return "generate"

    logger.info(
        "Context graded POOR (retry %s/%s) - routing to rewrite",
        state["retry_count"],
        state["max_retries"],
    )
    return "rewrite"
