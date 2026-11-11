"""Deterministic text primitives used across indexing, retrieval, and evaluation."""

from __future__ import annotations

import re
import unicodedata

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return " ".join(TOKEN_PATTERN.findall(normalized))


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(unicodedata.normalize("NFKC", text).lower())


def informative_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS and len(token) > 1]


def sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_PATTERN.split(text) if part.strip()]


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0
