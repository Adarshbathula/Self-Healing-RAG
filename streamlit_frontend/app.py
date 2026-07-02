import json
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


def auth_headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.token}"}


def register(email: str, password: str) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/auth/register", json={"email": email, "password": password}, timeout=10
    )
    response.raise_for_status()
    return response.json()


def login(email: str, password: str) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/auth/login", json={"email": email, "password": password}, timeout=10
    )
    response.raise_for_status()
    return response.json()


def upload_files(files: list) -> dict:
    payload = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files]
    response = requests.post(
        f"{API_BASE_URL}/upload", files=payload, headers=auth_headers(), timeout=300
    )
    response.raise_for_status()
    return response.json()


def ask_question_stream(question: str, meta_holder: dict):
    """Generator yielding answer tokens; fills meta_holder with sources/retries/quality."""
    response = requests.post(
        f"{API_BASE_URL}/ask/stream",
        json={"question": question},
        headers=auth_headers(),
        stream=True,
        timeout=120,
    )
    response.raise_for_status()

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        event = json.loads(line)
        event_type = event.get("type")
        if event_type == "meta":
            meta_holder.update(event)
        elif event_type == "token":
            yield event["text"]
        elif event_type == "error":
            raise RuntimeError(event.get("detail", "Streaming failed"))


def error_detail(exc: requests.RequestException) -> str:
    if exc.response is not None:
        try:
            return exc.response.json().get("detail", exc.response.text)
        except ValueError:
            return exc.response.text
    return str(exc)


if "token" not in st.session_state:
    st.session_state.token = None
    st.session_state.email = None
    st.session_state.role = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


def render_login_gate() -> None:
    st.markdown("## 🩹 Self-Healing RAG")
    st.caption("Sign in to continue. The first account ever registered becomes an admin automatically.")

    login_tab, register_tab = st.tabs(["Log in", "Register"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
        if submitted:
            try:
                result = login(email, password)
                st.session_state.token = result["access_token"]
                st.session_state.email = result["email"]
                st.session_state.role = result["role"]
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"Login failed: {error_detail(exc)}")

    with register_tab:
        with st.form("register_form"):
            email = st.text_input("Email", key="register_email")
            password = st.text_input(
                "Password", type="password", key="register_password", help="At least 8 characters"
            )
            submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
        if submitted:
            try:
                result = register(email, password)
                st.session_state.token = result["access_token"]
                st.session_state.email = result["email"]
                st.session_state.role = result["role"]
                st.success(f"Account created as {result['role']}")
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"Registration failed: {error_detail(exc)}")


if not st.session_state.token:
    render_login_gate()
    st.stop()

health = get_health()
is_admin = st.session_state.role == "admin"

with st.sidebar:
    st.markdown("### 🩹 Self-Healing RAG")
    st.caption(f"Signed in as **{st.session_state.email}** ({st.session_state.role})")

    if st.button("Log out", use_container_width=True):
        st.session_state.token = None
        st.session_state.email = None
        st.session_state.role = None
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**Knowledge base**")
    if health is None:
        st.error("Backend unreachable")
        st.code(API_BASE_URL, language=None)
    elif health["vector_store_ready"]:
        st.success(f"{health['indexed_documents']} chunks indexed")
    else:
        st.info("No documents indexed yet")

    if is_admin:
        with st.expander("📄 Upload PDFs", expanded=not (health and health.get("vector_store_ready"))):
            uploaded_files = st.file_uploader(
                "Add one or more PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )

            if st.button(
                "Upload & Index", disabled=not uploaded_files, type="primary", use_container_width=True
            ):
                with st.spinner("Uploading and indexing..."):
                    try:
                        result = upload_files(uploaded_files)
                        st.success(f"Added {result['total_chunks_added']} chunks")
                        for file_info in result["files"]:
                            st.caption(f"• {file_info['filename']} — {file_info['chunks_added']} chunks")
                        st.rerun()
                    except requests.RequestException as exc:
                        st.error(f"Upload failed: {error_detail(exc)}")
    else:
        st.caption("Only admins can upload documents.")

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
            st.warning("No documents indexed yet — ask an admin to upload one first, or ask a general question.")
        else:
            meta: dict = {}
            try:
                answer = st.write_stream(ask_question_stream(question, meta))
                sources = meta.get("sources", [])
                if sources:
                    with st.expander(f"📚 {len(sources)} source(s)"):
                        for source in sources:
                            st.markdown(f"**{source['source']}** · page {source['page']}")
                            st.caption(source["content"])
                            st.divider()
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except (requests.RequestException, RuntimeError) as exc:
                detail = error_detail(exc) if isinstance(exc, requests.RequestException) else str(exc)
                st.error(f"Failed to get an answer: {detail}")
