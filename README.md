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
streamlit run streamlit_app.py
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
