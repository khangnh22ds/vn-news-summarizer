"""HTML article extraction.

Primary path: :func:`trafilatura.extract` (best content extractor for
Vietnamese in benchmarks). Fallback: :class:`readability.Document`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

import trafilatura
from bs4 import BeautifulSoup
from loguru import logger
from readability import Document
from vn_news_common.text import normalize_text, word_count
from vn_news_common.time_utils import to_utc


@dataclass(slots=True)
class ExtractedArticle:
    title: str | None
    author: str | None
    published_at: datetime | None
    language: str | None
    content_text: str
    word_count: int


def extract_from_html(html: str, *, url: str | None = None) -> ExtractedArticle | None:
    """Extract the main article content from raw HTML.

    Returns ``None`` if both extractors fail to produce non-trivial text.
    """
    if not html:
        return None

    # Primary: trafilatura.
    try:
        extracted = trafilatura.extract(
            html,
            url=url,
            with_metadata=True,
            output_format="json",
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
        if extracted:
            doc = json.loads(extracted)
            text = normalize_text(doc.get("text") or "")
            if word_count(text) >= 50:
                return ExtractedArticle(
                    title=(doc.get("title") or None),
                    author=(doc.get("author") or None),
                    published_at=to_utc(doc.get("date")),
                    language=doc.get("language") or None,
                    content_text=text,
                    word_count=word_count(text),
                )
    except (ValueError, TypeError) as exc:
        logger.debug("trafilatura failed for {}: {}", url, exc)

    # Fallback: readability-lxml.
    try:
        doc = Document(html)
        title = doc.short_title()
        soup = BeautifulSoup(doc.summary(), "lxml")
        text = normalize_text(soup.get_text(" "))
        if word_count(text) >= 50:
            return ExtractedArticle(
                title=title or None,
                author=None,
                published_at=None,
                language=None,
                content_text=text,
                word_count=word_count(text),
            )
    except Exception as exc:
        logger.debug("readability failed for {}: {}", url, exc)

    return None
