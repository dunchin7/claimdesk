"""App-level wiring tests. These don't hit OpenAI."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_healthz() -> None:
    with TestClient(app) as client:
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_extract_validates_input() -> None:
    """Empty / too-short input should 422 before the LLM is called."""
    with TestClient(app) as client:
        r = client.post("/api/claims/extract", json={"raw_input": "x"})
        assert r.status_code == 422
