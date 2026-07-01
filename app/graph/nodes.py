from app.core.logging import get_logger
from app.graph.state import GraphState
from app.services.answer_generator import generate_answer
from app.services.context_grader import grade_context
from app.services.query_rewriter import rewrite_query
from app.services.retriever import build_sources, format_context, retrieve_documents

logger = get_logger(__name__)


def retrieve_node(state: GraphState) -> GraphState:
    query = state["active_question"]
    documents = retrieve_documents(query)

    return {
        **state,
        "retrieved_documents": documents,
        "retrieved_context": format_context(documents),
        "sources": build_sources(documents),
    }


def grade_context_node(state: GraphState) -> GraphState:
    result = grade_context(state["original_question"], state["retrieved_context"])
    return {**state, "context_quality": result.verdict}


def rewrite_query_node(state: GraphState) -> GraphState:
    rewritten = rewrite_query(state["original_question"], state["active_question"])
    return {
        **state,
        "rewritten_question": rewritten,
        "active_question": rewritten,
        "retry_count": state["retry_count"] + 1,
    }


def generate_answer_node(state: GraphState) -> GraphState:
    answer = generate_answer(state["original_question"], state["retrieved_context"])

    sources = state["sources"] if state["context_quality"] == "GOOD" else []
    return {**state, "answer": answer, "sources": sources}
