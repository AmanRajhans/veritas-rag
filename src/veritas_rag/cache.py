"""Index-aware SQLite response cache."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

from veritas_rag.models import QueryResponse
from veritas_rag.text import normalize_text


class ResponseCache:
    def __init__(self, path: Path, ttl_seconds: int = 3_600) -> None:
        self.path = path
        self.ttl_seconds = ttl_seconds
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS responses "
                "(cache_key TEXT PRIMARY KEY, created_at REAL NOT NULL, payload TEXT NOT NULL)"
            )

    def make_key(self, fingerprint: str, query: str, top_k: int) -> str:
        material = f"{fingerprint}\0{normalize_text(query)}\0{top_k}"
        return hashlib.sha256(material.encode()).hexdigest()

    def get(self, key: str) -> QueryResponse | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT created_at, payload FROM responses WHERE cache_key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            if time.time() - float(row[0]) > self.ttl_seconds:
                connection.execute("DELETE FROM responses WHERE cache_key = ?", (key,))
                return None
            return QueryResponse.model_validate_json(str(row[1]))

    def put(self, key: str, response: QueryResponse) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO responses(cache_key, created_at, payload) VALUES (?, ?, ?)",
                (key, time.time(), response.model_dump_json()),
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)
