"""Smoke test for the placeholder FastAPI app."""

from __future__ import annotations

from fastapi.testclient import TestClient
from vn_news_api import app


def test_healthz() -> None:
    client = TestClient(app)
    res = client.get("/healthz")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["version"] == app.version


def test_root() -> None:
    client = TestClient(app)
    res = client.get("/")
    assert res.status_code == 200
    body = res.json()
    assert body["name"] == "vn-news-summarizer"
    assert body["health"] == "/healthz"
    assert body["summarize"].startswith("POST")


def test_healthz_includes_model_id() -> None:
    client = TestClient(app)
    body = client.get("/healthz").json()
    assert "model_id" in body
    assert body["model_id"]  # non-empty string
