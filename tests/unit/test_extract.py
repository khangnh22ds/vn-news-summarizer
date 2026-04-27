"""Unit tests for the HTML article extractor."""

from __future__ import annotations

from pathlib import Path

from vn_news_crawler.extract import extract_from_html


def test_extract_returns_none_for_empty() -> None:
    assert extract_from_html("") is None


def test_extract_real_vietnamese_article(fixtures_dir: Path) -> None:
    html = (fixtures_dir / "sample_article.html").read_text(encoding="utf-8")
    res = extract_from_html(html, url="https://example.com/sea-games-33.html")
    assert res is not None
    assert res.word_count >= 50
    assert "Việt Nam" in res.content_text
    # Title should mention SEA Games (either trafilatura or readability path).
    assert res.title is not None
    assert "SEA Games" in res.title or "Việt Nam" in res.title
