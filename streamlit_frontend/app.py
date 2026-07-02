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

CUSTOM_CSS = """
<style>
#MainMenu, footer, header {visibility: hidden;}

:root {
    --teal-dark: #11998e;
    --teal-light: #38ef7d;
    --panel-dark: #1c1f2b;
    --panel-border: #2a2e3f;
}

/* ---- Login / register split card ---- */
.st-key-login_card {
    max-width: 900px;
    margin: 48px auto 0;
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 25px 60px rgba(17, 153, 142, 0.25);
}
.st-key-login_card [data-testid="stHorizontalBlock"] { gap: 0; }
.st-key-login_card [data-testid="column"] { padding: 0 !important; }

.login-left {
    background: linear-gradient(135deg, var(--teal-dark), var(--teal-light));
    color: white;
    min-height: 520px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 40px 32px;
}
.login-left .brand {
    font-size: 13px;
    letter-spacing: 2px;
    text-transform: uppercase;
    opacity: 0.85;
    margin-bottom: 48px;
}
.login-left h1 { font-size: 30px; margin: 0 0 14px; font-weight: 700; }
.login-left p { font-size: 14.5px; opacity: 0.92; max-width: 260px; line-height: 1.5; margin: 0; }

.st-key-login_right_panel {
    background: var(--panel-dark);
    border-left: 1px solid var(--panel-border);
    min-height: 520px;
    padding: 48px 40px 32px;
}

/* ---- Account avatar dropdown (top right, Gmail-style) ---- */
.st-key-account_menu button {
    border-radius: 50% !important;
    width: 42px !important;
    height: 42px !important;
    padding: 0 !important;
    background: linear-gradient(135deg, var(--teal-dark), var(--teal-light)) !important;
    color: white !important;
    font-weight: 700 !important;
    border: none !important;
    float: right;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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


def change_password(current_password: str, new_password: str) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/auth/change-password",
        json={"current_password": current_password, "new_password": new_password},
        headers=auth_headers(),
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def change_email(new_email: str, current_password: str) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/auth/change-email",
        json={"new_email": new_email, "current_password": current_password},
        headers=auth_headers(),
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


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
    with st.container(key="login_card"):
        left, right = st.columns([1, 1.3], gap="small")

        with left:
            st.markdown(
                """
                <div class="login-left">
                    <div class="brand">🩹 Self-Healing RAG</div>
                    <h1>Welcome Back!</h1>
                    <p>Sign in to ask questions grounded in your documents — powered by a
                    self-correcting retrieval pipeline that rewrites its own queries until it
                    finds a good answer.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right:
            with st.container(key="login_right_panel"):
                login_tab, register_tab = st.tabs(["Sign In", "Create Account"])

                with login_tab:
                    with st.form("login_form"):
                        email = st.text_input("Email")
                        password = st.text_input("Password", type="password")
                        submitted = st.form_submit_button(
                            "Sign In", type="primary", use_container_width=True
                        )
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
                            "Password",
                            type="password",
                            key="register_password",
                            help="At least 8 characters",
                        )
                        submitted = st.form_submit_button(
                            "Sign Up", type="primary", use_container_width=True
                        )
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
first_name = st.session_state.email.split("@")[0].replace(".", " ").title()

top_left, top_right = st.columns([8, 1])
with top_left:
    st.markdown(f"### 👋 Welcome back, {first_name}!")
    st.caption(
        "Ask anything below — we'll search your documents first, and fall back to general "
        "knowledge whenever they don't have the answer."
    )
with top_right:
    with st.container(key="account_menu"):
        with st.popover(st.session_state.email[0].upper()):
            st.markdown(f"**{st.session_state.email}**")
            st.caption(f"Role: {st.session_state.role}")
            st.divider()

            with st.expander("Change password"):
                with st.form("change_password_form"):
                    current_pw = st.text_input("Current password", type="password")
                    new_pw = st.text_input(
                        "New password", type="password", help="At least 8 characters"
                    )
                    pw_submitted = st.form_submit_button("Update password", use_container_width=True)
                if pw_submitted:
                    try:
                        change_password(current_pw, new_pw)
                        st.success("Password updated")
                    except requests.RequestException as exc:
                        st.error(error_detail(exc))

            with st.expander("Change email"):
                with st.form("change_email_form"):
                    new_email = st.text_input("New email")
                    confirm_pw = st.text_input("Current password", type="password", key="email_confirm_pw")
                    email_submitted = st.form_submit_button("Update email", use_container_width=True)
                if email_submitted:
                    try:
                        result = change_email(new_email, confirm_pw)
                        st.session_state.token = result["access_token"]
                        st.session_state.email = result["email"]
                        st.success("Email updated")
                        st.rerun()
                    except requests.RequestException as exc:
                        st.error(error_detail(exc))

            st.divider()
            if st.button("Sign out", use_container_width=True):
                st.session_state.token = None
                st.session_state.email = None
                st.session_state.role = None
                st.session_state.messages = []
                st.rerun()

with st.sidebar:
    st.markdown("### 🩹 Self-Healing RAG")
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
