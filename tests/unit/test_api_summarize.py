"""Unit tests for ``POST /summarize`` (no real model load, no network).

We stub the summarizer singleton with a fake whose ``summarize`` is
deterministic and instant. URL-mode tests stub the article fetcher
likewise so the test suite never touches the network.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from vn_news_api import app
from vn_news_api.routers import summarize as summarize_router
from vn_news_api.services.article_fetcher import ArticleFetchError, FetchedArticle
from vn_news_common.settings import get_settings, reset_settings


class _FakeSummarizer:
    """Records the last input and returns a marker summary."""

    def __init__(self, marker: str = "FAKE_SUMMARY") -> None:
        self.marker = marker
        self.last_input: str | None = None

    def summarize(self, text: str) -> str:
        self.last_input = text
        return f"{self.marker}:{len(text)}"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, _FakeSummarizer]:
    fake = _FakeSummarizer()
    monkeypatch.setattr(summarize_router, "get_summarizer", lambda: fake)
    return TestClient(app), fake


def test_summarize_text_mode_happy_path(
    client: tuple[TestClient, _FakeSummarizer],
) -> None:
    tc, fake = client
    body = "Bài báo tiếng Việt với nhiều câu để tóm tắt." * 3
    res = tc.post("/summarize", json={"text": body})
    assert res.status_code == 200
    payload = res.json()
    assert payload["summary"].startswith("FAKE_SUMMARY:")
    assert payload["source_url"] is None
    assert payload["source_title"] is None
    assert payload["input_chars"] == len(body)
    assert payload["summary_chars"] == len(payload["summary"])
    assert payload["elapsed_ms"] >= 0
    assert payload["model_id"]
    assert fake.last_input == body


def test_summarize_text_mode_strips_whitespace(
    client: tuple[TestClient, _FakeSummarizer],
) -> None:
    tc, fake = client
    res = tc.post("/summarize", json={"text": "   abc   "})
    assert res.status_code == 200
    assert fake.last_input == "abc"


def test_summarize_url_mode_calls_fetcher(
    monkeypatch: pytest.MonkeyPatch,
    client: tuple[TestClient, _FakeSummarizer],
) -> None:
    tc, fake = client

    captured: dict[str, Any] = {}

    def _fake_fetch(url: str) -> FetchedArticle:
        captured["url"] = url
        return FetchedArticle(
            title="Bài báo mẫu",
            content_text="Nội dung bài báo bằng tiếng Việt.",
            word_count=6,
        )

    monkeypatch.setattr(summarize_router, "fetch_and_extract", _fake_fetch)
    res = tc.post("/summarize", json={"url": "https://example.com/article"})
    assert res.status_code == 200
    payload = res.json()
    # HttpUrl normalises a trailing slash; loosen the comparison.
    assert payload["source_url"].startswith("https://example.com/article")
    assert payload["source_title"] == "Bài báo mẫu"
    assert payload["input_chars"] == len("Nội dung bài báo bằng tiếng Việt.")
    assert fake.last_input == "Nội dung bài báo bằng tiếng Việt."
    assert captured["url"].startswith("https://example.com/article")


def test_summarize_url_mode_propagates_fetch_error(
    monkeypatch: pytest.MonkeyPatch,
    client: tuple[TestClient, _FakeSummarizer],
) -> None:
    tc, _ = client

    def _boom(url: str) -> FetchedArticle:
        del url
        raise ArticleFetchError("upstream_error", "upstream returned HTTP 503")

    monkeypatch.setattr(summarize_router, "fetch_and_extract", _boom)
    res = tc.post("/summarize", json={"url": "https://example.com/article"})
    assert res.status_code == 422
    detail = res.json()["detail"]
    assert detail["reason"] == "upstream_error"


def test_summarize_rejects_neither_input(
    client: tuple[TestClient, _FakeSummarizer],
) -> None:
    tc, _ = client
    res = tc.post("/summarize", json={})
    assert res.status_code == 422


def test_summarize_rejects_both_inputs(
    client: tuple[TestClient, _FakeSummarizer],
) -> None:
    tc, _ = client
    res = tc.post(
        "/summarize",
        json={"text": "có nội dung", "url": "https://example.com/x"},
    )
    assert res.status_code == 422


def test_summarize_rejects_blank_text(
    client: tuple[TestClient, _FakeSummarizer],
) -> None:
    tc, _ = client
    res = tc.post("/summarize", json={"text": "   "})
    assert res.status_code == 422


def test_summarize_truncates_long_input(
    monkeypatch: pytest.MonkeyPatch,
    client: tuple[TestClient, _FakeSummarizer],
) -> None:
    """If the body exceeds ``summarize_max_input_chars``, the head is kept."""
    tc, fake = client

    reset_settings()
    monkeypatch.setenv("SUMMARIZE_MAX_INPUT_CHARS", "100")
    cap = get_settings().summarize_max_input_chars
    body = "x" * (cap + 500)
    res = tc.post("/summarize", json={"text": body})
    assert res.status_code == 200
    assert fake.last_input is not None
    assert len(fake.last_input) == cap

    reset_settings()
