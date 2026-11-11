"""Offline retrieval and answer evaluation without an external judge model."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from veritas_rag.pipeline import VeritasPipeline
from veritas_rag.text import tokenize


@dataclass(frozen=True, slots=True)
class EvalCase:
    query: str
    relevant_sections: list[str]
    reference_answer: str


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        try:
            cases.append(
                EvalCase(
                    query=str(raw["query"]),
                    relevant_sections=[str(value) for value in raw["relevant_sections"]],
                    reference_answer=str(raw["reference_answer"]),
                )
            )
        except KeyError as error:
            raise ValueError(f"line {line_number} is missing {error.args[0]}") from error
    return cases


def evaluate(pipeline: VeritasPipeline, cases: list[EvalCase], top_k: int = 5) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        hits = pipeline.retriever.search(case.query, top_k=top_k)
        relevant = {value.lower() for value in case.relevant_sections}
        relevance = [hit.chunk.section.lower() in relevant for hit in hits]
        found = {hit.chunk.section.lower() for hit in hits} & relevant
        reciprocal_rank = next(
            (1 / rank for rank, value in enumerate(relevance, start=1) if value), 0.0
        )
        dcg = sum(
            (1.0 if value else 0.0) / math.log2(rank + 1) for rank, value in enumerate(relevance, 1)
        )
        ideal_count = min(len(relevant), top_k)
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        response = pipeline.query(case.query, top_k=top_k)
        rows.append(
            {
                "query": case.query,
                "recall_at_k": len(found) / max(len(relevant), 1),
                "mrr": reciprocal_rank,
                "ndcg_at_k": dcg / ideal_dcg if ideal_dcg else 0.0,
                "answer_token_f1": token_f1(response.answer, case.reference_answer),
                "abstained": response.abstained,
                "latency_ms": response.latency_ms,
            }
        )
    metric_names = ["recall_at_k", "mrr", "ndcg_at_k", "answer_token_f1", "latency_ms"]
    summary = {
        name: mean(float(row[name]) for row in rows) if rows else 0.0 for name in metric_names
    }
    summary["abstention_rate"] = mean(float(row["abstained"]) for row in rows) if rows else 0.0
    summary["cases"] = len(rows)
    return {"summary": summary, "cases": rows}


def token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = tokenize(prediction)
    reference_tokens = tokenize(reference)
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    prediction_counts = {token: prediction_tokens.count(token) for token in set(prediction_tokens)}
    reference_counts = {token: reference_tokens.count(token) for token in set(reference_tokens)}
    overlap = sum(
        min(count, reference_counts.get(token, 0)) for token, count in prediction_counts.items()
    )
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
