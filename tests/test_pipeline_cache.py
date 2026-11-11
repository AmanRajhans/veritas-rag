from __future__ import annotations

from pathlib import Path

from veritas_rag.cache import ResponseCache
from veritas_rag.index import CorpusIndex
from veritas_rag.pipeline import VeritasPipeline


def test_pipeline_returns_grounded_citations(pipeline: VeritasPipeline) -> None:
    response = pipeline.query("When does airfare require manager approval?", include_trace=True)
    assert not response.abstained
    assert "[" in response.answer
    assert response.citations
    assert response.citations[0].source == "policy.md"
    assert response.trace is not None


def test_pipeline_abstains_on_out_of_domain_question(pipeline: VeritasPipeline) -> None:
    response = pipeline.query("Who won the lunar chess championship on Mars?")
    assert response.abstained
    assert response.citations == []


def test_cache_is_index_and_query_aware(sample_index: CorpusIndex, tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache.sqlite3")
    pipeline = VeritasPipeline(sample_index, cache=cache, confidence_threshold=0.25)
    first = pipeline.query("What happens to break-glass access?", include_trace=True)
    second = pipeline.query("What happens to break-glass access?", include_trace=True)
    assert first.answer == second.answer
    assert second.trace is not None and second.trace.cache_hit
    other_key = cache.make_key("another-index", "What happens to break-glass access?", 5)
    assert cache.get(other_key) is None
