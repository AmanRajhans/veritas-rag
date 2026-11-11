from __future__ import annotations

from veritas_rag.evaluation import EvalCase, evaluate, token_f1
from veritas_rag.pipeline import VeritasPipeline


def test_token_f1() -> None:
    assert token_f1("alpha beta", "alpha beta") == 1.0
    assert token_f1("alpha", "beta") == 0.0
    assert token_f1("", "") == 1.0


def test_evaluation_report(pipeline: VeritasPipeline) -> None:
    report = evaluate(
        pipeline,
        [
            EvalCase(
                query="When is manager approval required for airfare?",
                relevant_sections=["Expense Policy"],
                reference_answer="Airfare always requires manager approval.",
            )
        ],
        top_k=3,
    )
    assert report["summary"]["recall_at_k"] == 1.0
    assert report["summary"]["mrr"] == 1.0
    assert report["summary"]["cases"] == 1
