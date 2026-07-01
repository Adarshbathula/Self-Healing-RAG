# Self-Healing RAG

A production-ready Retrieval-Augmented Generation service orchestrated end-to-end by **LangGraph**.
Retrieval quality is graded automatically; if context is poor, the query is rewritten and retrieval
is retried until context is good or a retry limit is reached.

## Stack

- FastAPI (async)
- LangGraph `StateGraph` workflow
- LangChain + FAISS vector store (persisted to disk)
- **Groq** (`langchain-groq`) for the LLM — grading, query rewriting, and answer generation
- **HuggingFace `sentence-transformers`** (local, free) for embeddings — Groq has no embeddings API,
  so `BAAI/bge-small-en-v1.5` runs locally via `langchain-huggingface`
- Pydantic v2 / pydantic-settings

## Workflow

```
START -> Retrieve -> Grade Context --GOOD--> Generate -> END
                          |
                        POOR
                          |
                          v
                    Rewrite Query -> Retrieve (loop until GOOD or max_retries)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Copy `.env` and set your `GROQ_API_KEY` (get one at https://console.groq.com/keys):

```env
GROQ_API_KEY=gsk_xxx
MODEL_NAME=llama-3.3-70b-versatile
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
FAISS_INDEX_PATH=faiss_index
DOCUMENTS_DIR=documents
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=4
MAX_RETRIES=3
```

## Run

Start the API:

```bash
uvicorn main:app --reload
```

Docs at `http://localhost:8000/docs`.

In a second terminal, start the Streamlit UI:

```bash
streamlit run streamlit_frontend/app.py
```

UI at `http://localhost:8501`. Set `API_BASE_URL` if the API isn't on `http://localhost:8000`.

## Endpoints

### `POST /upload`
`multipart/form-data`, field name `files` (one or more PDFs). Extracts text, chunks it, embeds it,
and adds it to the persisted FAISS index.

### `POST /ask`
```json
{ "question": "What is Retrieval-Augmented Generation?" }
```

```json
{
  "answer": "...",
  "sources": [{ "page": 5, "source": "paper.pdf", "content": "..." }],
  "retries": 1,
  "context_quality": "GOOD"
}
```

### `GET /health`
```json
{ "status": "ok", "vector_store_ready": true, "indexed_documents": 128 }
```

## Deploying (Render + Streamlit Community Cloud)

The backend (FastAPI + LangGraph) and frontend (Streamlit) deploy as two separate services.

### 1. Backend on Render

1. Push this repo to GitHub (already done if you're reading this from the repo).
2. On [render.com](https://render.com) → **New +** → **Blueprint** → connect this GitHub repo. Render
   will detect [render.yaml](render.yaml) and pre-fill the service (build command, start command,
   health check, env var names).
   - Prefer manual setup instead? **New +** → **Web Service**, runtime **Python 3**, build command
     `pip install -r requirements.txt`, start command `uvicorn main:app --host 0.0.0.0 --port $PORT`.
3. In the service's **Environment** tab, set `GROQ_API_KEY` to your real key (never commit it — it's
   the one value `render.yaml` marks `sync: false` so Render prompts you for it instead of reading
   the repo). The other vars (`MODEL_NAME`, `EMBEDDING_MODEL`, chunk sizes, etc.) already come from
   `render.yaml`.
4. Deploy. First build takes 5-10 minutes — it installs `torch`/`sentence-transformers`/`faiss` and
   downloads the embedding model on first startup.
5. Once live, verify: `curl https://<your-service>.onrender.com/health`.

**Persistence caveat:** without a mounted disk, anything written to the local filesystem (uploaded
PDFs, the FAISS index) is lost on every restart/redeploy. `render.yaml` requests a 1 GB persistent
disk at `storage/`, mapped via `FAISS_INDEX_PATH=storage/faiss_index` and
`DOCUMENTS_DIR=storage/documents` — but **persistent disks require a paid plan** (`plan: starter` in
the blueprint, ~$7/mo+the disk). If you deploy on the free plan, remove the `disk:` block from
`render.yaml` and expect to re-upload your PDFs after the service sleeps/restarts.

**Resource caveat:** the embedding model (`sentence-transformers` + `torch`) is memory-hungry. Render's
free tier (512 MB RAM) may OOM or be very slow loading it — the `starter` plan or higher is recommended
for reliable operation.

### 1b. Backend on Hugging Face Spaces (free alternative to Render)

Render's free tier has only 512 MB RAM (too little for `torch`/`sentence-transformers` reliably) and
persistent disks require a paid plan. Hugging Face Spaces' free CPU tier gives 16 GB RAM, no card
required, and supports Docker — a better fit for this app at zero cost.

This repo ships a [Dockerfile](Dockerfile) (listens on port 7860, per HF's convention) and a dedicated
`huggingface-space` branch whose `README.md` has the YAML frontmatter Spaces needs
(`sdk: docker`, `app_port: 7860`) without cluttering the GitHub-facing README on `main`.

1. On [huggingface.co](https://huggingface.co) → **New Space** → SDK: **Docker** → pick a name (e.g.
   `self-healing-rag`) → visibility your choice.
2. In the Space's **Settings → Repository secrets**, add `GROQ_API_KEY` with your real key.
3. Locally, add the Space as a git remote and push the `huggingface-space` branch to its `main`:
   ```bash
   git remote add hf https://huggingface.co/spaces/<your-username>/self-healing-rag
   git push hf huggingface-space:main
   ```
   (Hugging Face will ask you to log in — use a [HF access token](https://huggingface.co/settings/tokens)
   as the password when prompted.)
4. The Space builds the Docker image automatically (~5-10 min first time) and serves the API at
   `https://<your-username>-self-healing-rag.hf.space`.
5. Verify: `curl https://<your-username>-self-healing-rag.hf.space/health`.

**Persistence caveat still applies**: HF Spaces' free tier storage is also ephemeral by default —
uploaded PDFs and the FAISS index reset on a rebuild/restart unless you add persistent storage
(paid, Settings → persistent storage). Whenever you update `main`, re-sync the frontmatter branch:
`git checkout huggingface-space && git merge main && git push hf huggingface-space:main`.

### 2. Frontend on Streamlit Community Cloud

The Streamlit app lives in its own directory ([streamlit_frontend/](streamlit_frontend/)) with its
**own slim `requirements.txt`** (just `streamlit` + `requests`) — this matters because Streamlit
Community Cloud installs whatever `requirements.txt` sits next to the entrypoint file, and you don't
want it pulling in `torch`/`faiss`/`langchain` just to run the UI.

1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick this GitHub repo and
   branch `main`.
2. Set **Main file path** to `streamlit_frontend/app.py`.
3. Open **Advanced settings** → **Secrets**, and add:
   ```toml
   API_BASE_URL = "https://<your-render-service>.onrender.com"
   ```
   (Streamlit Cloud exposes secrets.toml keys as environment variables too, which is what
   `os.environ.get("API_BASE_URL")` in the app reads.)
4. Deploy. The backend already sends permissive CORS headers (`allow_origins=["*"]` in
   [main.py](main.py)), so the cross-origin call from `*.streamlit.app` to Render works out of the box.

## Betterment suggestions (beyond the base spec)

- **Groq instead of OpenAI**: `ChatGroq` gives very low-latency inference, which matters a lot here
  since a single `/ask` call can invoke the LLM 2-4x (grader, rewriter(s), generator) across retries.
- **Local embeddings**: since Groq doesn't serve an embeddings endpoint, embeddings run locally via
  `sentence-transformers` — zero embedding API cost/latency and no external dependency for ingestion.
- **Structured grading with fallback**: the grader uses `with_structured_output`, but falls back to a
  `PydanticOutputParser` chain if the model doesn't support tool-based structured output reliably.
- **Deduplication in context formatting**: retrieved chunks are deduplicated before being sent to the
  grader/generator to avoid inflating "GOOD" verdicts on redundant chunks.
- **Thread-safe singleton vector store** with disk persistence so the index survives restarts and
  concurrent `/upload` calls don't race on `save_local`.
- **Global exception handler + per-route try/except** so a single failed ingestion or generation
  doesn't leak stack traces to clients.
- Consider adding: request-level rate limiting, an auth layer (API key/JWT) before exposing `/upload`
  publicly, and swapping `FAISS` for a managed vector DB (e.g. Qdrant/pgvector) if you need multi-tenant
  isolation or horizontal scaling beyond a single-node index.
