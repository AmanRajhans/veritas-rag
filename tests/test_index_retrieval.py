from __future__ import annotations

from pathlib import Path

import pytest

from veritas_rag.index import CorpusIndex
from veritas_rag.retrieval import HybridRetriever


def test_index_round_trip(sample_index: CorpusIndex, tmp_path: Path) -> None:
    path = tmp_path / "nested" / "index.json"
    sample_index.save(path)
    loaded = CorpusIndex.load(path)
    assert loaded.fingerprint == sample_index.fingerprint
    assert loaded.chunks == sample_index.chunks


def test_index_detects_tampering(sample_index: CorpusIndex, tmp_path: Path) -> None:
    path = tmp_path / "index.json"
    sample_index.save(path)
    path.write_text(path.read_text().replace("manager approval", "secret override", 1))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        CorpusIndex.load(path)


def test_hybrid_retrieval_ranks_relevant_section_first(sample_index: CorpusIndex) -> None:
    hits = HybridRetriever(sample_index).search("When does airfare need manager approval?", top_k=3)
    assert hits[0].chunk.section == "Expense Policy"
    assert hits[0].score > hits[-1].score
    assert all(0 <= hit.semantic_score <= 1 for hit in hits)


def test_empty_query_returns_no_results(sample_index: CorpusIndex) -> None:
    assert HybridRetriever(sample_index).search("   ") == []
