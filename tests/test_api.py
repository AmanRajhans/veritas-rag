from __future__ import annotations

from fastapi.testclient import TestClient

from veritas_rag.api import create_app
from veritas_rag.pipeline import VeritasPipeline


def test_health_and_query(pipeline: VeritasPipeline) -> None:
    with TestClient(create_app(pipeline)) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        response = client.post(
            "/v1/query",
            json={"query": "When does airfare require approval?", "top_k": 3},
        )
        assert response.status_code == 200
        assert response.json()["citations"]

        retrieval = client.post("/v1/retrieve", json={"query": "canary rollback", "top_k": 2})
        assert retrieval.status_code == 200
        assert retrieval.json()[0]["section"] == "Deployments"
        assert "veritas_requests_total" in client.get("/metrics").text


def test_missing_index_is_degraded(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VERITAS_INDEX_PATH", str(tmp_path / "missing.json"))
    with TestClient(create_app()) as client:
        assert client.get("/healthz").json()["status"] == "degraded"
        response = client.post("/v1/query", json={"query": "valid question"})
        assert response.status_code == 503
