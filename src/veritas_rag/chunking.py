"""Markdown-aware ingestion and overlapping chunk construction."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from veritas_rag.models import Chunk

SUPPORTED_SUFFIXES = {".md", ".txt", ".json", ".jsonl"}
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True, slots=True)
class SourceDocument:
    text: str
    source: str
    title: str


def load_documents(path: Path) -> list[SourceDocument]:
    """Load supported files in stable path order without executing embedded content."""
    files = (
        [path]
        if path.is_file()
        else sorted(item for item in path.rglob("*") if item.suffix.lower() in SUPPORTED_SUFFIXES)
    )
    documents: list[SourceDocument] = []
    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix in {".md", ".txt"}:
            text = file_path.read_text(encoding="utf-8")
            title = _first_heading(text) or file_path.stem.replace("-", " ").title()
            documents.append(SourceDocument(text=text, source=str(file_path), title=title))
        elif suffix == ".json":
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            records = payload if isinstance(payload, list) else [payload]
            documents.extend(_documents_from_records(records, file_path))
        elif suffix == ".jsonl":
            records = [
                json.loads(line)
                for line in file_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            documents.extend(_documents_from_records(records, file_path))
    return documents


def _documents_from_records(records: list[object], path: Path) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    for position, raw in enumerate(records):
        if not isinstance(raw, dict) or "text" not in raw:
            raise ValueError(f"{path}: record {position} must be an object containing 'text'")
        source = str(raw.get("source", f"{path}#{position + 1}"))
        title = str(raw.get("title", path.stem.replace("-", " ").title()))
        documents.append(SourceDocument(text=str(raw["text"]), source=source, title=title))
    return documents


def chunk_documents(
    documents: list[SourceDocument], target_words: int = 140, overlap_words: int = 30
) -> list[Chunk]:
    if target_words < 20:
        raise ValueError("target_words must be at least 20")
    if not 0 <= overlap_words < target_words:
        raise ValueError("overlap_words must be non-negative and smaller than target_words")

    chunks: list[Chunk] = []
    for document in documents:
        sections = _split_sections(document.text, document.title)
        ordinal = 0
        carry: list[str] = []
        for section, body in sections:
            words = body.split()
            if not words:
                continue
            start = 0
            while start < len(words):
                prefix = carry if start == 0 else []
                room = max(1, target_words - len(prefix))
                current = prefix + words[start : start + room]
                text = " ".join(current).strip()
                if text:
                    digest = hashlib.sha256(
                        f"{document.source}\0{section}\0{ordinal}\0{text}".encode()
                    ).hexdigest()[:16]
                    chunks.append(
                        Chunk(
                            chunk_id=digest,
                            text=text,
                            source=document.source,
                            title=document.title,
                            section=section,
                            ordinal=ordinal,
                        )
                    )
                    ordinal += 1
                    carry = current[-overlap_words:] if overlap_words else []
                start += room
    return chunks


def _split_sections(text: str, fallback_title: str) -> list[tuple[str, str]]:
    current = fallback_title
    buffer: list[str] = []
    sections: list[tuple[str, str]] = []
    for line in text.splitlines():
        heading = HEADING_PATTERN.match(line)
        if heading:
            if any(part.strip() for part in buffer):
                sections.append((current, "\n".join(buffer).strip()))
            current = heading.group(2).strip()
            buffer = []
        elif line.strip():
            buffer.append(line.strip())
    if any(part.strip() for part in buffer):
        sections.append((current, "\n".join(buffer).strip()))
    return sections


def _first_heading(text: str) -> str | None:
    for line in text.splitlines():
        match = HEADING_PATTERN.match(line)
        if match:
            return match.group(2).strip()
    return None
