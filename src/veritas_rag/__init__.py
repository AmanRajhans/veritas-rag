"""Veritas RAG: evidence-first retrieval augmented generation."""

from veritas_rag.index import CorpusIndex
from veritas_rag.pipeline import VeritasPipeline

__all__ = ["CorpusIndex", "VeritasPipeline"]
__version__ = "0.1.0"
