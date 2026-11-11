"""End-to-end RAG orchestration and calibrated abstention."""

from __future__ import annotations

import time
import uuid

from veritas_rag.cache import ResponseCache
from veritas_rag.generation import (
    AnswerGenerator,
    ExtractiveGenerator,
    materialize_citations,
)
from veritas_rag.index import CorpusIndex
from veritas_rag.models import QueryResponse, RetrievalTrace
from veritas_rag.retrieval import HybridRetriever
from veritas_rag.text import informative_tokens


class VeritasPipeline:
    def __init__(
        self,
        index: CorpusIndex,
        *,
        generator: AnswerGenerator | None = None,
        cache: ResponseCache | None = None,
        confidence_threshold: float = 0.33,
    ) -> None:
        self.index = index
        self.retriever = HybridRetriever(index)
        self.generator = generator or ExtractiveGenerator()
        self.cache = cache
        self.confidence_threshold = confidence_threshold

    def query(self, query: str, top_k: int = 5, include_trace: bool = False) -> QueryResponse:
        started = time.perf_counter()
        cache_key = (
            self.cache.make_key(self.index.fingerprint, query, top_k) if self.cache else None
        )
        if self.cache and cache_key:
            cached = self.cache.get(cache_key)
            if cached is not None:
                cached.latency_ms = round((time.perf_counter() - started) * 1_000, 3)
                if include_trace:
                    cached.trace = cached.trace or RetrievalTrace(
                        query_tokens=informative_tokens(query),
                        retrieved_chunk_ids=[citation.chunk_id for citation in cached.citations],
                        retrieval_scores=[
                            citation.retrieval_score for citation in cached.citations
                        ],
                        cache_hit=True,
                    )
                    cached.trace.cache_hit = True
                return cached

        evidence = self.retriever.search(query, top_k=top_k)
        generated = self.generator.generate(query, evidence)
        query_tokens = set(informative_tokens(query))
        evidence_tokens = (
            set().union(*(set(informative_tokens(item.chunk.text)) for item in evidence))
            if evidence
            else set()
        )
        coverage = len(query_tokens & evidence_tokens) / max(len(query_tokens), 1)
        top_score = evidence[0].score if evidence else 0.0
        citation_factor = min(len(generated.citation_ids) / 2, 1.0)
        confidence = min(max(0.52 * top_score + 0.30 * coverage + 0.18 * citation_factor, 0.0), 1.0)
        abstained = (
            confidence < self.confidence_threshold
            or not generated.text
            or generated.text == "INSUFFICIENT_EVIDENCE"
            or not generated.citation_ids
        )
        if abstained:
            answer = "I don't have enough grounded evidence in the indexed corpus to answer that."
            citations = []
        else:
            answer = generated.text
            citations = materialize_citations(evidence, generated.citation_ids)

        trace = None
        if include_trace:
            trace = RetrievalTrace(
                query_tokens=sorted(query_tokens),
                retrieved_chunk_ids=[item.chunk.chunk_id for item in evidence],
                retrieval_scores=[round(item.score, 6) for item in evidence],
            )
        response = QueryResponse(
            answer=answer,
            citations=citations,
            confidence=round(confidence, 6),
            abstained=abstained,
            latency_ms=round((time.perf_counter() - started) * 1_000, 3),
            trace_id=uuid.uuid4().hex,
            trace=trace,
        )
        if self.cache and cache_key:
            self.cache.put(cache_key, response)
        return response
