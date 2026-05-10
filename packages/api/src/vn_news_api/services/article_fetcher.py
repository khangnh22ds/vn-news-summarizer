"""Fetch a Vietnamese news article URL and extract its main body.

Reuses :func:`vn_news_crawler.extract.extract_from_html` (trafilatura
primary, readability fallback) so the on-demand /summarize path produces
the same cleaned text that the offline labeling pipeline saw at training
time. Network IO uses ``httpx`` rather than the rate-limited crawler
client because /summarize is user-initiated, not a bulk crawl.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from loguru import logger
from vn_news_common.settings import get_settings
from vn_news_crawler.extract import extract_from_html


class ArticleFetchError(RuntimeError):
    """Raised when fetching or extracting an article URL fails.

    The :attr:`reason` short-tag is suitable for the API's JSON error
    body so callers can react programmatically (e.g. distinguish a 404
    upstream from an unparseable HTML page).
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(slots=True)
class FetchedArticle:
    title: str | None
    content_text: str
    word_count: int


def fetch_and_extract(url: str) -> FetchedArticle:
    """Download ``url`` and return the extracted main body.

    Raises :class:`ArticleFetchError` for network errors, non-2xx
    responses, and pages whose extracted body is too short to be a real
    article (the crawler considers <50 words a failure).
    """
    settings = get_settings()
    timeout = settings.summarize_url_timeout_seconds
    headers = {"User-Agent": settings.crawler_user_agent}

    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers=headers,
        ) as client:
            resp = client.get(url)
    except httpx.HTTPError as exc:
        logger.warning("article fetch network error for {}: {}", url, exc)
        msg = f"could not fetch URL: {exc}"
        raise ArticleFetchError("fetch_failed", msg) from exc

    if resp.status_code >= 400:
        logger.warning("article fetch upstream {} for {}", resp.status_code, url)
        msg = f"upstream returned HTTP {resp.status_code}"
        raise ArticleFetchError("upstream_error", msg)

    extracted = extract_from_html(resp.text, url=url)
    if extracted is None or not extracted.content_text.strip():
        msg = "could not extract article body from URL"
        raise ArticleFetchError("extract_failed", msg)

    return FetchedArticle(
        title=extracted.title,
        content_text=extracted.content_text,
        word_count=extracted.word_count,
    )
