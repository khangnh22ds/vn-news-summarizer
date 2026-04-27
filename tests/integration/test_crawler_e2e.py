"""End-to-end pipeline test: mocked RSS + HTML → SQLite persistence."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import select
from vn_news_common.db import dispose_engine, get_engine, session_scope
from vn_news_common.models import Article, Base, CrawlRun, Source
from vn_news_common.settings import reset_settings
from vn_news_crawler.config import SourcesConfig
from vn_news_crawler.http_client import PoliteClient
from vn_news_crawler.pipeline import _crawl_one_source, run_once
from vn_news_crawler.robots import RobotsCache

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_HTML = (REPO_ROOT / "tests" / "fixtures" / "sample_article.html").read_text(
    encoding="utf-8"
)


def _build_rss(items: list[tuple[str, str]]) -> str:
    """Build a minimal RSS 2.0 feed from (title, link) tuples."""
    entries = "".join(
        f"<item><title>{title}</title><link>{link}</link>"
        f"<pubDate>Tue, 15 Apr 2025 08:30:00 +0700</pubDate></item>"
        for title, link in items
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<title>Mock Feed</title><link>https://news.example.com</link>
<description>test</description>
{entries}
</channel></rss>"""


def _make_handler(
    rss_xml: str,
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/robots.txt"):
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        if "feed" in url or url.endswith(".rss"):
            return httpx.Response(
                200,
                content=rss_xml.encode("utf-8"),
                headers={"Content-Type": "application/rss+xml"},
            )
        return httpx.Response(200, text=FIXTURE_HTML)

    return handler


def _make_pipeline_clients(rss_xml: str) -> tuple[PoliteClient, RobotsCache]:
    """Build PoliteClient + RobotsCache backed by an httpx MockTransport."""
    transport = httpx.MockTransport(_make_handler(rss_xml))
    polite_client = httpx.AsyncClient(transport=transport, headers={"User-Agent": "test/0.1"})
    robots_client = httpx.AsyncClient(transport=transport, headers={"User-Agent": "test/0.1"})
    polite = PoliteClient(
        user_agent="test/0.1",
        timeout_s=5.0,
        rps_per_host=0.0,
        max_retries=1,
        client=polite_client,
    )
    robots = RobotsCache(user_agent="test/0.1", client=robots_client)
    return polite, robots


@pytest_asyncio.fixture
async def isolated_db(tmp_path: Path) -> AsyncIterator[None]:
    db_path = tmp_path / "e2e.db"
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path}"
    reset_settings()
    await dispose_engine()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await dispose_engine()
    os.environ.pop("DATABASE_URL", None)
    reset_settings()


def _build_cfg(rss_url: str) -> SourcesConfig:
    return SourcesConfig.model_validate(
        {
            "defaults": {
                "user_agent": "test/0.1",
                "crawl_delay_s": 0.0,
                "timeout_s": 5,
                "max_retries": 1,
            },
            "sources": [
                {
                    "id": "mocksrc",
                    "name": "Mock Source",
                    "domain": "news.example.com",
                    "enabled": True,
                    "rss": [rss_url],
                }
            ],
            "canonical_categories": {"the_thao": ["the-thao", "thể thao"]},
        }
    )


@pytest.mark.asyncio
async def test_pipeline_persists_articles(isolated_db: None) -> None:
    rss_xml = _build_rss(
        [
            ("Bài 1", "https://news.example.com/the-thao/bai-1.html"),
            ("Bài 2", "https://news.example.com/the-thao/bai-2.html"),
            ("Bài 3", "https://news.example.com/the-thao/bai-3.html"),
        ]
    )
    cfg = _build_cfg("https://news.example.com/feed.rss")
    src = cfg.sources[0]

    polite, robots = _make_pipeline_clients(rss_xml)
    try:
        stats = await _crawl_one_source(src=src, cfg=cfg, http=polite, robots=robots)
    finally:
        await polite.aclose()
        await robots.aclose()

    assert stats.discovered == 3
    # All 3 share the same body text → simhash dedupe collapses 2 of 3.
    assert stats.inserted == 1
    assert stats.skipped_dupe == 2

    async with session_scope() as s:
        rows = (await s.execute(select(Article))).scalars().all()
        assert len(rows) == 1
        article = rows[0]
        assert article.word_count >= 50
        assert article.simhash is not None
        sources = (await s.execute(select(Source))).scalars().all()
        assert len(sources) == 1
        assert sources[0].source_id == "mocksrc"


@pytest.mark.asyncio
async def test_pipeline_skips_disallowed_robots(isolated_db: None) -> None:
    rss_xml = _build_rss([("Blocked", "https://news.example.com/private/secret.html")])
    cfg = _build_cfg("https://news.example.com/feed.rss")
    src = cfg.sources[0]

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/robots.txt"):
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
        if "feed" in url:
            return httpx.Response(
                200,
                content=rss_xml.encode("utf-8"),
                headers={"Content-Type": "application/rss+xml"},
            )
        return httpx.Response(200, text=FIXTURE_HTML)

    transport = httpx.MockTransport(handler)
    polite_client = httpx.AsyncClient(transport=transport, headers={"User-Agent": "test/0.1"})
    robots_client = httpx.AsyncClient(transport=transport, headers={"User-Agent": "test/0.1"})
    polite = PoliteClient(
        user_agent="test/0.1",
        timeout_s=5.0,
        rps_per_host=0.0,
        max_retries=1,
        client=polite_client,
    )
    robots = RobotsCache(user_agent="test/0.1", client=robots_client)

    try:
        stats = await _crawl_one_source(src=src, cfg=cfg, http=polite, robots=robots)
    finally:
        await polite.aclose()
        await robots.aclose()

    assert stats.skipped_robots == 1
    assert stats.inserted == 0


@pytest.mark.asyncio
async def test_run_once_records_crawl_run(
    isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    rss_xml = _build_rss([("Bài duy nhất", "https://news.example.com/the-thao/x.html")])
    cfg = _build_cfg("https://news.example.com/feed.rss")
    transport = httpx.MockTransport(_make_handler(rss_xml))
    original = httpx.AsyncClient

    def patched(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs.setdefault("transport", transport)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("vn_news_crawler.http_client.httpx.AsyncClient", patched)
    monkeypatch.setattr("vn_news_crawler.robots.httpx.AsyncClient", patched)

    report = await run_once(cfg)
    assert report.total_inserted == 1

    async with session_scope() as s:
        runs = (await s.execute(select(CrawlRun))).scalars().all()
        assert len(runs) == 1
        assert runs[0].n_new == 1
        assert runs[0].notes is not None and "inserted=1" in runs[0].notes
