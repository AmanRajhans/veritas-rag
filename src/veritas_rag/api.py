"""FastAPI service with health, retrieval, query, and lightweight metrics endpoints."""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

from veritas_rag.cache import ResponseCache
from veritas_rag.index import CorpusIndex
from veritas_rag.models import QueryRequest, QueryResponse, RetrievalHit
from veritas_rag.pipeline import VeritasPipeline


def create_app(pipeline: VeritasPipeline | None = None) -> FastAPI:
    counters = {"requests": 0, "errors": 0, "latency_ms": 0.0}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if pipeline is not None:
            app.state.pipeline = pipeline
        else:
            index_path = Path(os.getenv("VERITAS_INDEX_PATH", "artifacts/demo-index.json"))
            if index_path.exists():
                cache_path = Path(os.getenv("VERITAS_CACHE_PATH", "artifacts/cache.sqlite3"))
                app.state.pipeline = VeritasPipeline(
                    CorpusIndex.load(index_path), cache=ResponseCache(cache_path)
                )
            else:
                app.state.pipeline = None
        yield

    application = FastAPI(
        title="Veritas RAG API",
        version="0.1.0",
        description="Grounded hybrid retrieval with explicit citations and abstention.",
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def measure(request: Request, call_next):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        counters["requests"] += 1
        try:
            return await call_next(request)
        except Exception:
            counters["errors"] += 1
            raise
        finally:
            counters["latency_ms"] += (time.perf_counter() - started) * 1_000

    def require_pipeline(request: Request) -> VeritasPipeline:
        current = request.app.state.pipeline
        if current is None:
            raise HTTPException(status_code=503, detail="index not loaded")
        return current

    @application.get("/healthz")
    def health(request: Request) -> dict[str, object]:
        current = request.app.state.pipeline
        return {
            "status": "ok" if current is not None else "degraded",
            "index_fingerprint": current.index.fingerprint if current else None,
            "chunks": len(current.index.chunks) if current else 0,
        }

    @application.post("/v1/query", response_model=QueryResponse)
    def query(payload: QueryRequest, request: Request) -> QueryResponse:
        return require_pipeline(request).query(
            payload.query, top_k=payload.top_k, include_trace=payload.include_trace
        )

    @application.post("/v1/retrieve", response_model=list[RetrievalHit])
    def retrieve(payload: QueryRequest, request: Request) -> list[RetrievalHit]:
        return [
            RetrievalHit(
                chunk_id=hit.chunk.chunk_id,
                source=hit.chunk.source,
                title=hit.chunk.title,
                section=hit.chunk.section,
                text=hit.chunk.text,
                score=hit.score,
                bm25_score=hit.bm25_score,
                semantic_score=hit.semantic_score,
            )
            for hit in require_pipeline(request).retriever.search(payload.query, payload.top_k)
        ]

    @application.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        requests = int(counters["requests"])
        average = counters["latency_ms"] / requests if requests else 0.0
        return (
            "# TYPE veritas_requests_total counter\n"
            f"veritas_requests_total {requests}\n"
            "# TYPE veritas_errors_total counter\n"
            f"veritas_errors_total {int(counters['errors'])}\n"
            "# TYPE veritas_request_latency_ms gauge\n"
            f"veritas_request_latency_ms {average:.3f}\n"
        )

    return application


app = create_app()
