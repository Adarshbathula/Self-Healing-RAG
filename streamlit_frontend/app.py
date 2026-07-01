import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Self-Healing RAG", page_icon="🩹", layout="wide")

EXAMPLE_QUESTIONS = [
    "Summarize the uploaded documents",
    "What are the key skills mentioned?",
    "What is Retrieval-Augmented Generation?",
]


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


def error_detail(exc: requests.RequestException) -> str:
    if exc.response is not None:
        try:
            return exc.response.json().get("detail", exc.response.text)
        except ValueError:
            return exc.response.text
    return str(exc)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

health = get_health()

with st.sidebar:
    st.markdown("### 🩹 Self-Healing RAG")
    st.caption("Ask questions grounded in your documents — falls back to general knowledge when they don't cover it.")

    st.divider()
    st.markdown("**Knowledge base**")
    if health is None:
        st.error("Backend unreachable")
        st.code(API_BASE_URL, language=None)
    elif health["vector_store_ready"]:
        st.success(f"{health['indexed_documents']} chunks indexed")
    else:
        st.info("No documents indexed yet")

    with st.expander("📄 Upload PDFs", expanded=not (health and health.get("vector_store_ready"))):
        uploaded_files = st.file_uploader(
            "Add one or more PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if st.button("Upload & Index", disabled=not uploaded_files, type="primary", use_container_width=True):
            with st.spinner("Uploading and indexing..."):
                try:
                    result = upload_files(uploaded_files)
                    st.success(f"Added {result['total_chunks_added']} chunks")
                    for file_info in result["files"]:
                        st.caption(f"• {file_info['filename']} — {file_info['chunks_added']} chunks")
                    st.rerun()
                except requests.RequestException as exc:
                    st.error(f"Upload failed: {error_detail(exc)}")

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True, disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.rerun()

st.markdown("## Ask a question")

if not st.session_state.messages:
    st.caption("Try one of these, or ask your own question below:")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, example in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(example, use_container_width=True):
            st.session_state.pending_question = example

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"📚 {len(message['sources'])} source(s)"):
                for source in message["sources"]:
                    st.markdown(f"**{source['source']}** · page {source['page']}")
                    st.caption(source["content"])
                    st.divider()

typed_question = st.chat_input("Ask about your documents, or anything else...")
question = typed_question or st.session_state.pending_question
st.session_state.pending_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        if health is None:
            st.error("Backend unreachable — make sure the FastAPI server is running.")
        elif not health.get("vector_store_ready"):
            st.warning("No documents indexed yet — upload a PDF from the sidebar first, or ask a general question.")
        else:
            with st.spinner("Thinking..."):
                try:
                    result = ask_question(question)
                    st.write(result["answer"])
                    if result["sources"]:
                        with st.expander(f"📚 {len(result['sources'])} source(s)"):
                            for source in result["sources"]:
                                st.markdown(f"**{source['source']}** · page {source['page']}")
                                st.caption(source["content"])
                                st.divider()
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"],
                        }
                    )
                except requests.RequestException as exc:
                    st.error(f"Failed to get an answer: {error_detail(exc)}")
