from functools import lru_cache

from langgraph.graph import END, StateGraph

from app.core.config import get_settings
from app.core.logging import get_logger
from app.graph.nodes import (
    generate_answer_node,
    grade_context_node,
    retrieve_node,
    rewrite_query_node,
)
from app.graph.router import route_after_grading
from app.graph.state import GraphState

logger = get_logger(__name__)


def _add_retrieval_loop(graph: StateGraph, generate_target: str) -> None:
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_context", grade_context_node)
    graph.add_node("rewrite_query", rewrite_query_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_context")
    graph.add_conditional_edges(
        "grade_context",
        route_after_grading,
        {"generate": generate_target, "rewrite": "rewrite_query"},
    )
    graph.add_edge("rewrite_query", "retrieve")


def build_workflow():
    """Full graph: retrieve -> grade -> (rewrite loop) -> generate -> END."""
    graph = StateGraph(GraphState)
    _add_retrieval_loop(graph, generate_target="generate")
    graph.add_node("generate", generate_answer_node)
    graph.add_edge("generate", END)
    return graph.compile()


def build_context_workflow():
    """Retrieval-only graph: retrieve -> grade -> (rewrite loop) -> END.

    Stops right before answer generation so the caller can stream the final
    answer separately instead of waiting for the whole graph to finish.
    """
    graph = StateGraph(GraphState)
    _add_retrieval_loop(graph, generate_target=END)
    return graph.compile()


@lru_cache
def get_compiled_workflow():
    logger.info("Compiling self-healing RAG LangGraph workflow")
    return build_workflow()


@lru_cache
def get_compiled_context_workflow():
    logger.info("Compiling context-only LangGraph workflow (for streaming answers)")
    return build_context_workflow()


def _build_initial_state(question: str) -> GraphState:
    settings = get_settings()
    return {
        "original_question": question,
        "rewritten_question": "",
        "active_question": question,
        "retrieved_documents": [],
        "retrieved_context": "",
        "answer": "",
        "retry_count": 0,
        "max_retries": settings.max_retries,
        "context_quality": "",
        "sources": [],
    }


async def run_workflow(question: str) -> GraphState:
    workflow = get_compiled_workflow()
    return await workflow.ainvoke(_build_initial_state(question))


async def prepare_context(question: str) -> GraphState:
    """Runs the retrieve/grade/rewrite loop only, without generating an answer."""
    workflow = get_compiled_context_workflow()
    return await workflow.ainvoke(_build_initial_state(question))
