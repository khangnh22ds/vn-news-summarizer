"""RSS / Atom feed reader.

Returns a list of :class:`ArticleCandidate` per feed. We use ``feedparser``
because it tolerates the wide variety of malformed feeds we see in the
wild.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import feedparser
from loguru import logger
from vn_news_common.schemas import ArticleCandidate
from vn_news_common.time_utils import to_utc

from .http_client import PoliteClient

_CATEGORY_FROM_PATH_RE = re.compile(r"^/?([\w\-]+)/", re.IGNORECASE)


def _category_from_url(url: str) -> str | None:
    """Heuristic: take the first path segment of the canonical URL.

    Falls back to ``None`` if nothing meaningful is found.
    """
    try:
        path = url.split("//", 1)[1].split("/", 1)[1]
    except IndexError:
        return None
    m = _CATEGORY_FROM_PATH_RE.match(path)
    if m:
        return m.group(1).replace("-", "_").lower()
    return None


def _parse_published(entry: Any) -> datetime | None:
    raw = (
        entry.get("published")
        or entry.get("updated")
        or entry.get("pubDate")
        or entry.get("dc_date")
    )
    return to_utc(raw)


async def fetch_feed(
    client: PoliteClient,
    *,
    source_id: str,
    feed_url: str,
) -> list[ArticleCandidate]:
    """Fetch a single RSS/Atom feed and return article candidates."""
    try:
        res = await client.get(feed_url)
    except Exception as exc:
        logger.error("RSS fetch failed for {}: {}", feed_url, exc)
        return []

    parsed = feedparser.parse(res.content)
    if parsed.bozo and not parsed.entries:
        logger.warning("RSS parse error for {}: {}", feed_url, parsed.bozo_exception)
        return []

    out: list[ArticleCandidate] = []
    for entry in parsed.entries:
        link = entry.get("link") or ""
        title = (entry.get("title") or "").strip()
        if not link or not title:
            continue
        rss_category = None
        if entry.get("tags"):
            rss_category = entry["tags"][0].get("term") or None
        out.append(
            ArticleCandidate(
                source_id=source_id,
                url=link,
                title=title,
                published_at=_parse_published(entry),
                rss_category=rss_category or _category_from_url(link),
                author=(entry.get("author") or None),
            )
        )
    return out
