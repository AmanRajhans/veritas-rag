from __future__ import annotations

import json
from pathlib import Path

import pytest

from veritas_rag.chunking import SourceDocument, chunk_documents, load_documents


def test_chunking_preserves_section_metadata_and_is_deterministic() -> None:
    documents = [SourceDocument("# One\nalpha beta gamma delta", "a.md", "A")]
    first = chunk_documents(documents, target_words=20, overlap_words=2)
    second = chunk_documents(documents, target_words=20, overlap_words=2)
    assert first == second
    assert first[0].section == "One"
    assert len(first[0].chunk_id) == 16


def test_chunking_validates_window() -> None:
    with pytest.raises(ValueError, match="target_words"):
        chunk_documents([], target_words=10)
    with pytest.raises(ValueError, match="overlap_words"):
        chunk_documents([], target_words=20, overlap_words=20)


def test_load_jsonl_documents(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({"title": "A", "text": "hello", "source": "custom"}) + "\n")
    documents = load_documents(path)
    assert documents[0].title == "A"
    assert documents[0].source == "custom"


def test_invalid_json_record_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"title":"missing text"}')
    with pytest.raises(ValueError, match="containing 'text'"):
        load_documents(path)
