from __future__ import annotations

import json
from pathlib import Path

from veritas_rag.cli import main


def test_index_ask_and_evaluate_commands(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "guide.md").write_text(
        "# Access\nBreak-glass access expires after four hours and creates an alert."
    )
    index_path = tmp_path / "index.json"

    assert main(["index", str(source), "--output", str(index_path)]) == 0
    index_output = json.loads(capsys.readouterr().out)
    assert index_output["documents"] == 1
    assert index_path.exists()

    assert (
        main(
            [
                "ask",
                "How long does break-glass access last?",
                "--index",
                str(index_path),
                "--no-cache",
                "--trace",
            ]
        )
        == 0
    )
    answer = json.loads(capsys.readouterr().out)
    assert answer["abstained"] is False
    assert answer["trace"] is not None

    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "query": "When does break-glass access expire?",
                "relevant_sections": ["Access"],
                "reference_answer": "It expires after four hours.",
            }
        )
        + "\n"
    )
    report_path = tmp_path / "reports" / "eval.json"
    assert (
        main(
            [
                "evaluate",
                str(dataset),
                "--index",
                str(index_path),
                "--output",
                str(report_path),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["summary"]["recall_at_k"] == 1.0
    assert report_path.exists()
