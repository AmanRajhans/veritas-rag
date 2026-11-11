"""Shared domain and API models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True, slots=True)
class Chunk:
    """A stable, source-addressable unit of evidence."""

    chunk_id: str
    text: str
    source: str
    title: str
    section: str
    ordinal: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "title": self.title,
            "section": self.section,
            "ordinal": self.ordinal,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Chunk:
        return cls(
            chunk_id=str(value["chunk_id"]),
            text=str(value["text"]),
            source=str(value["source"]),
            title=str(value["title"]),
            section=str(value["section"]),
            ordinal=int(value["ordinal"]),
        )


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    bm25_score: float
    semantic_score: float
    rank: int


class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=3, max_length=1_000)
    top_k: int = Field(default=5, ge=1, le=20)
    include_trace: bool = False


class Citation(BaseModel):
    citation_id: int
    chunk_id: str
    source: str
    title: str
    section: str
    quote: str
    retrieval_score: float


class RetrievalTrace(BaseModel):
    query_tokens: list[str]
    retrieved_chunk_ids: list[str]
    retrieval_scores: list[float]
    cache_hit: bool = False


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float = Field(ge=0.0, le=1.0)
    abstained: bool
    latency_ms: float
    trace_id: str
    trace: RetrievalTrace | None = None


class RetrievalHit(BaseModel):
    chunk_id: str
    source: str
    title: str
    section: str
    text: str
    score: float
    bm25_score: float
    semantic_score: float
