import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Self-Healing RAG", page_icon="🩹", layout="wide")


def get_health() -> dict | None:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def upload_files(files: list) -> dict:
    payload = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files]
    response = requests.post(f"{API_BASE_URL}/upload", files=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def ask_question(question: str) -> dict:
    response = requests.post(f"{API_BASE_URL}/ask", json={"question": question}, timeout=120)
    response.raise_for_status()
    return response.json()


st.title("🩹 Self-Healing RAG")
st.caption("LangGraph-orchestrated retrieval that grades its own context and rewrites the query until it's good.")

health = get_health()
with st.sidebar:
    st.header("Status")
    if health is None:
        st.error("Backend unreachable")
        st.code(API_BASE_URL)
    elif health["vector_store_ready"]:
        st.success(f"Ready — {health['indexed_documents']} chunks indexed")
    else:
        st.warning("No documents indexed yet")

    st.divider()
    st.header("1. Upload documents")
    uploaded_files = st.file_uploader(
        "Add one or more PDFs", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Upload & Index", disabled=not uploaded_files, type="primary"):
        with st.spinner("Uploading and indexing..."):
            try:
                result = upload_files(uploaded_files)
                st.success(f"{result['message']} — total chunks added: {result['total_chunks_added']}")
                for file_info in result["files"]:
                    st.write(f"• {file_info['filename']}: {file_info['chunks_added']} chunks")
            except requests.RequestException as exc:
                detail = getattr(exc.response, "text", str(exc)) if exc.response is not None else str(exc)
                st.error(f"Upload failed: {detail}")

st.header("2. Ask a question")
st.caption("Retrieval runs, context is graded, and the query is rewritten automatically if the context is poor.")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Question", placeholder="e.g. What is Retrieval-Augmented Generation?")
ask_clicked = st.button("Ask", type="primary", disabled=not question)

if ask_clicked and question:
    if health is None or not health.get("vector_store_ready"):
        st.error("No documents have been ingested yet. Upload PDFs first.")
    else:
        with st.spinner("Running self-healing retrieval workflow..."):
            try:
                result = ask_question(question)
                st.session_state.history.insert(0, {"question": question, "result": result})
            except requests.RequestException as exc:
                detail = getattr(exc.response, "text", str(exc)) if exc.response is not None else str(exc)
                st.error(f"Failed to get an answer: {detail}")

for entry in st.session_state.history:
    result = entry["result"]
    with st.container(border=True):
        st.markdown(f"**Q: {entry['question']}**")

        quality = result["context_quality"]
        badge_color = "green" if quality == "GOOD" else "red"
        st.markdown(
            f":{badge_color}[Context: {quality}]  &nbsp;&nbsp;  Retries: {result['retries']}"
        )

        st.write(result["answer"])

        if result["sources"]:
            with st.expander(f"Sources ({len(result['sources'])})"):
                for source in result["sources"]:
                    st.markdown(f"**{source['source']} — page {source['page']}**")
                    st.text(source["content"])
                    st.divider()
