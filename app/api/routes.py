import json

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from app.api.deps import get_current_user, require_admin
from app.core.logging import get_logger
from app.graph.workflow import prepare_context, run_workflow
from app.models.request import AskRequest
from app.models.response import (
    AskResponse,
    HealthResponse,
    SourceChunk,
    UploadedFile,
    UploadResponse,
)
from app.services.answer_generator import stream_answer
from app.services.ingestion import ingest_uploaded_file
from app.services.vector_store import get_vector_store_manager

logger = get_logger(__name__)

router = APIRouter()


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_documents(
    files: list[UploadFile], _admin: dict = Depends(require_admin)
) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided")

    uploaded: list[UploadedFile] = []
    total_chunks = 0

    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Only PDF files are supported, got: {file.filename}",
            )

        try:
            chunks_added = await ingest_uploaded_file(file)
        except Exception:
            logger.exception("Failed to ingest file %s", file.filename)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process file: {file.filename}",
            ) from None

        uploaded.append(UploadedFile(filename=file.filename, chunks_added=chunks_added))
        total_chunks += chunks_added

    return UploadResponse(
        message=f"Successfully ingested {len(uploaded)} file(s)",
        files=uploaded,
        total_chunks_added=total_chunks,
    )


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest, _user: dict = Depends(get_current_user)) -> AskResponse:
    manager = get_vector_store_manager()
    if not manager.is_ready:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents have been ingested yet. Upload PDFs via /upload first.",
        )

    try:
        final_state = await run_workflow(request.question)
    except Exception:
        logger.exception("Workflow execution failed for question: %s", request.question)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate an answer",
        ) from None

    sources = [SourceChunk(**source) for source in final_state["sources"]]

    return AskResponse(
        answer=final_state["answer"],
        sources=sources,
        retries=final_state["retry_count"],
        context_quality=final_state["context_quality"],
    )


@router.post("/ask/stream")
async def ask_question_stream(
    request: AskRequest, _user: dict = Depends(get_current_user)
) -> StreamingResponse:
    manager = get_vector_store_manager()
    if not manager.is_ready:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents have been ingested yet. Upload PDFs via /upload first.",
        )

    try:
        context_state = await prepare_context(request.question)
    except Exception:
        logger.exception("Context retrieval failed for question: %s", request.question)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve context",
        ) from None

    sources = context_state["sources"] if context_state["context_quality"] == "GOOD" else []
    meta = {
        "type": "meta",
        "sources": sources,
        "retries": context_state["retry_count"],
        "context_quality": context_state["context_quality"],
    }

    async def event_stream():
        yield json.dumps(meta) + "\n"
        try:
            async for token in stream_answer(request.question, context_state["retrieved_context"]):
                yield json.dumps({"type": "token", "text": token}) + "\n"
        except Exception:
            logger.exception("Streaming answer generation failed for question: %s", request.question)
            yield json.dumps({"type": "error", "detail": "Failed to generate an answer"}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    manager = get_vector_store_manager()
    return HealthResponse(
        status="ok",
        vector_store_ready=manager.is_ready,
        indexed_documents=manager.document_count,
    )
