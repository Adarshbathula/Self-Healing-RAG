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


def build_workflow():
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_context", grade_context_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("generate", generate_answer_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_context")
    graph.add_conditional_edges(
        "grade_context",
        route_after_grading,
        {"generate": "generate", "rewrite": "rewrite_query"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()


@lru_cache
def get_compiled_workflow():
    logger.info("Compiling self-healing RAG LangGraph workflow")
    return build_workflow()


async def run_workflow(question: str) -> GraphState:
    settings = get_settings()
    workflow = get_compiled_workflow()

    initial_state: GraphState = {
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

    final_state = await workflow.ainvoke(initial_state)
    return final_state
