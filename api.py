"""FastAPI backend exposing the RAG pipeline + feedback collection +
on-demand re-ingestion.

Endpoints:
    POST /api/rag-query  -- answer a question (optionally with conversation history)
    POST /api/feedback   -- log thumbs up/down + optional comment
    POST /api/ingest     -- rebuild the dense + BM25 indices from data/ folder
    GET  /health         -- liveness probe + counters

Run with:
    uvicorn api:app --port 8000
"""

import json
import os
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from logger_config import configure_logging, get_logger
from rag import RAGEngine

load_dotenv()
configure_logging()
log = get_logger("api")

FEEDBACK_LOG = Path("feedback.jsonl")

# Simple in-process metrics — exposable to Prometheus later.
metrics = {
    "requests": 0,
    "errors": 0,
    "feedback_up": 0,
    "feedback_down": 0,
    "ingestions": 0,
}

# Optional auth token for the ingest endpoint. If set in .env, callers must
# provide it as the X-Admin-Token header. If unset, the endpoint is open
# (fine for local dev, gate it in production).
INGEST_ADMIN_TOKEN = os.getenv("INGEST_ADMIN_TOKEN", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("startup")
    app.state.rag = RAGEngine()
    yield
    log.info("shutdown")


app = FastAPI(
    title="Discord RAG FAQ Bot — Backend",
    description="RAG-powered backend for the AI Bootcamp Discord FAQ chatbot.",
    version="0.3.0",
    lifespan=lifespan,
)


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=4000)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    user_id: Optional[str] = None
    history: Optional[list[HistoryMessage]] = Field(
        default=None,
        description="Optional prior conversation turns for follow-up questions.",
    )


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: int
    request_id: str


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: Literal["up", "down"]
    user_id: Optional[str] = None
    comment: Optional[str] = None


class IngestResponse(BaseModel):
    ok: bool
    chunks: int
    duration_ms: int
    message: str


@app.middleware("http")
async def request_logger(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    log.info(
        "request_in",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )
    try:
        response = await call_next(request)
    except Exception:
        metrics["errors"] += 1
        log.exception("request_error", request_id=request_id)
        raise
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    log.info(
        "request_out",
        request_id=request_id,
        status=response.status_code,
        latency_ms=elapsed_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "metrics": metrics}


@app.post("/api/rag-query", response_model=QueryResponse)
async def rag_query(req: QueryRequest, request: Request) -> QueryResponse:
    metrics["requests"] += 1
    start = time.perf_counter()
    try:
        history_dicts = (
            [{"role": m.role, "content": m.content} for m in req.history]
            if req.history
            else None
        )
        result = app.state.rag.answer(req.query, history=history_dicts)
    except Exception as e:
        metrics["errors"] += 1
        log.exception("rag_failure", query=req.query)
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {e}")

    latency_ms = int((time.perf_counter() - start) * 1000)
    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        latency_ms=latency_ms,
        request_id=request.headers.get("X-Request-ID", ""),
    )


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest) -> dict:
    if req.rating == "up":
        metrics["feedback_up"] += 1
    else:
        metrics["feedback_down"] += 1

    record = {
        "ts": time.time(),
        "user_id": req.user_id,
        "rating": req.rating,
        "query": req.query,
        "answer": req.answer,
        "comment": req.comment,
    }
    with FEEDBACK_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    log.info("feedback_recorded", rating=req.rating, user_id=req.user_id)
    return {"ok": True}


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest(request: Request) -> IngestResponse:
    """Re-run the ingestion pipeline against the current `data/` folder.

    Use this after dropping new docs into `data/` to refresh the KB without
    a full redeploy. Spawns `ingest.py` as a subprocess (isolated env, fresh
    process) and reloads the in-process RAGEngine so subsequent queries see
    the new indices.

    Auth: if INGEST_ADMIN_TOKEN is set in .env, callers must pass it as the
    `X-Admin-Token` header. Otherwise the endpoint is open (suitable for
    local dev only).
    """
    if INGEST_ADMIN_TOKEN:
        provided = request.headers.get("X-Admin-Token", "")
        if provided != INGEST_ADMIN_TOKEN:
            log.warning("ingest_unauthorized")
            raise HTTPException(status_code=401, detail="Invalid admin token")

    log.info("ingest_triggered")
    start = time.perf_counter()

    # Run ingest.py in a subprocess. This isolates dependency state and means
    # if the script fails, the API doesn't crash — we just return 500.
    try:
        result = subprocess.run(
            [sys.executable, "ingest.py"],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes — generous for large KBs
            cwd=Path(__file__).parent,
        )
    except subprocess.TimeoutExpired:
        log.exception("ingest_timeout")
        raise HTTPException(status_code=504, detail="Ingest timed out after 5 minutes")

    if result.returncode != 0:
        log.error(
            "ingest_subprocess_failed",
            returncode=result.returncode,
            stderr=result.stderr[-2000:],
        )
        raise HTTPException(
            status_code=500,
            detail=f"Ingest subprocess failed: {result.stderr[-500:]}",
        )

    # Hot-reload the in-process engine so the new indices take effect without
    # restarting the server.
    try:
        app.state.rag = RAGEngine()
    except Exception as e:
        log.exception("rag_reload_failed")
        raise HTTPException(
            status_code=500,
            detail=f"Indices rebuilt but reload failed: {e}",
        )

    duration_ms = int((time.perf_counter() - start) * 1000)
    chunks = len(app.state.rag.chunks)
    metrics["ingestions"] += 1
    log.info("ingest_complete", chunks=chunks, duration_ms=duration_ms)
    return IngestResponse(
        ok=True,
        chunks=chunks,
        duration_ms=duration_ms,
        message=f"Re-ingested {chunks} chunks from data/ in {duration_ms} ms.",
    )
