"""Hybrid retrieval, reciprocal-rank fusion, transparent reranking, and diversity."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from veritas_rag.index import CorpusIndex
from veritas_rag.models import RetrievedChunk
from veritas_rag.text import informative_tokens, jaccard, tokenize


class HybridRetriever:
    def __init__(
        self,
        index: CorpusIndex,
        *,
        bm25_weight: float = 0.55,
        semantic_weight: float = 0.45,
        rrf_constant: int = 60,
        diversity: float = 0.82,
    ) -> None:
        self.index = index
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.rrf_constant = rrf_constant
        self.diversity = diversity

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        candidate_count = min(len(self.index.chunks), max(top_k * 4, 20))
        bm25 = self.index.bm25.scores(query)
        semantic = self.index.semantic.scores(query)
        bm25_order = _descending_indices(bm25, candidate_count)
        semantic_order = _descending_indices(semantic, candidate_count)

        fused: defaultdict[int, float] = defaultdict(float)
        for rank, index in enumerate(bm25_order, start=1):
            fused[index] += self.bm25_weight / (self.rrf_constant + rank)
        for rank, index in enumerate(semantic_order, start=1):
            fused[index] += self.semantic_weight / (self.rrf_constant + rank)

        if not fused:
            return []
        maximum_fused = max(fused.values()) or 1.0
        maximum_bm25 = max(float(bm25.max()), 1e-9)
        query_tokens = set(informative_tokens(query))
        scored: list[tuple[int, float]] = []
        for index, reciprocal_rank_score in fused.items():
            chunk = self.index.chunks[index]
            chunk_tokens = set(informative_tokens(f"{chunk.title} {chunk.section} {chunk.text}"))
            coverage = len(query_tokens & chunk_tokens) / max(len(query_tokens), 1)
            exact_phrase = float(" ".join(tokenize(query)) in " ".join(tokenize(chunk.text)))
            score = (
                0.48 * reciprocal_rank_score / maximum_fused
                + 0.27 * coverage
                + 0.15 * float(semantic[index])
                + 0.07 * float(bm25[index]) / maximum_bm25
                + 0.03 * exact_phrase
            )
            scored.append((index, min(max(score, 0.0), 1.0)))

        selected = self._diversify(scored, top_k)
        return [
            RetrievedChunk(
                chunk=self.index.chunks[index],
                score=score,
                bm25_score=float(bm25[index]),
                semantic_score=float(semantic[index]),
                rank=rank,
            )
            for rank, (index, score) in enumerate(selected, start=1)
        ]

    def _diversify(
        self, candidates: list[tuple[int, float]], top_k: int
    ) -> list[tuple[int, float]]:
        remaining = sorted(candidates, key=lambda item: (-item[1], item[0]))
        selected: list[tuple[int, float]] = []
        selected_tokens: list[set[str]] = []
        while remaining and len(selected) < top_k:
            best_position = 0
            best_mmr = float("-inf")
            for position, (index, relevance) in enumerate(remaining):
                tokens = set(tokenize(self.index.chunks[index].text))
                redundancy = max((jaccard(tokens, other) for other in selected_tokens), default=0.0)
                mmr = self.diversity * relevance - (1 - self.diversity) * redundancy
                if mmr > best_mmr:
                    best_position, best_mmr = position, mmr
            index, relevance = remaining.pop(best_position)
            selected.append((index, relevance))
            selected_tokens.append(set(tokenize(self.index.chunks[index].text)))
        return selected


def _descending_indices(scores: np.ndarray, limit: int) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))[:limit]
