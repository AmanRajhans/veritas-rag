"""Serializable lexical and local-semantic corpus index."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from veritas_rag.models import Chunk
from veritas_rag.text import tokenize

INDEX_VERSION = 1


class BM25Index:
    """A compact BM25 implementation with inspectable scoring."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(document) for document in documents]
        self.term_frequencies = [Counter(document) for document in self.tokens]
        self.lengths = np.asarray([len(document) for document in self.tokens], dtype=np.float64)
        self.average_length = float(self.lengths.mean()) if len(self.lengths) else 0.0
        document_frequency: Counter[str] = Counter()
        for document in self.tokens:
            document_frequency.update(set(document))
        size = len(self.tokens)
        self.idf = {
            term: math.log(1.0 + (size - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def scores(self, query: str) -> np.ndarray:
        result = np.zeros(len(self.tokens), dtype=np.float64)
        if not self.tokens or self.average_length == 0:
            return result
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, frequencies in enumerate(self.term_frequencies):
                frequency = frequencies.get(term, 0)
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * self.lengths[index] / self.average_length
                )
                if denominator:
                    result[index] += idf * frequency * (self.k1 + 1) / denominator
        return result


class LocalSemanticIndex:
    """Character n-gram TF-IDF retrieval for typo and morphology robustness."""

    def __init__(self, documents: list[str]) -> None:
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_features=30_000,
            sublinear_tf=True,
            lowercase=True,
        )
        self.matrix = normalize(self.vectorizer.fit_transform(documents), norm="l2")

    def scores(self, query: str) -> np.ndarray:
        query_vector = normalize(self.vectorizer.transform([query]), norm="l2")
        return np.asarray((self.matrix @ query_vector.T).toarray(), dtype=np.float64).ravel()


class CorpusIndex:
    """In-memory search index rebuilt from a safe, portable JSON artifact."""

    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("cannot build an index without chunks")
        self.chunks = chunks
        documents = [f"{chunk.title} {chunk.section} {chunk.text}" for chunk in chunks]
        self.bm25 = BM25Index(documents)
        self.semantic = LocalSemanticIndex(documents)
        canonical = json.dumps([chunk.to_dict() for chunk in chunks], sort_keys=True)
        self.fingerprint = hashlib.sha256(canonical.encode()).hexdigest()[:20]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": INDEX_VERSION,
            "fingerprint": self.fingerprint,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CorpusIndex:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != INDEX_VERSION:
            raise ValueError(
                f"unsupported index version {payload.get('version')}; expected {INDEX_VERSION}"
            )
        index = cls([Chunk.from_dict(item) for item in payload["chunks"]])
        if payload.get("fingerprint") != index.fingerprint:
            raise ValueError("index fingerprint mismatch; artifact may be corrupted")
        return index
