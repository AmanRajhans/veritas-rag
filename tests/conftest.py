from __future__ import annotations

from pathlib import Path

import pytest

from veritas_rag.chunking import SourceDocument, chunk_documents
from veritas_rag.index import CorpusIndex
from veritas_rag.pipeline import VeritasPipeline


@pytest.fixture
def sample_documents() -> list[SourceDocument]:
    return [
        SourceDocument(
            source="policy.md",
            title="Policy",
            text=(
                "# Expense Policy\n"
                "Expenses under $100 need no approval. Expenses from $100 to $500 require "
                "manager approval before purchase. Airfare always requires manager approval.\n"
                "# Security\nProduction access expires after ninety days. Break-glass access "
                "expires after four hours and creates an alert."
            ),
        ),
        SourceDocument(
            source="engineering.md",
            title="Engineering",
            text=(
                "# Deployments\nA canary receives five percent of traffic for ten minutes. "
                "The system rolls back if error rate exceeds two percent."
            ),
        ),
    ]


@pytest.fixture
def sample_index(sample_documents: list[SourceDocument]) -> CorpusIndex:
    return CorpusIndex(chunk_documents(sample_documents, target_words=40, overlap_words=5))


@pytest.fixture
def pipeline(sample_index: CorpusIndex) -> VeritasPipeline:
    return VeritasPipeline(sample_index, confidence_threshold=0.25)


@pytest.fixture
def temp_index(sample_index: CorpusIndex, tmp_path: Path) -> Path:
    path = tmp_path / "index.json"
    sample_index.save(path)
    return path
