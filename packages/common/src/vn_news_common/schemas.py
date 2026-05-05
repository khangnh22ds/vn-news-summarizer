"""Pydantic schemas for serialization (API in/out, IPC between services).

Kept separate from ORM models so we never accidentally leak DB-only
fields. These schemas are the canonical contract for the public API.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from vn_news_common.enums import ArticleStatus, LabelQuality, LabelSource


class _Base(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


# ------------------------------------------------------------------ Source


class SourceConfig(_Base):
    """Loaded from ``configs/sources.yaml``."""

    id: str
    name: str
    domain: str
    enabled: bool = True
    rss: list[HttpUrl] = Field(default_factory=list)
    crawl_delay_s: float | None = None
    timeout_s: float | None = None
    max_retries: int | None = None
    # Cap items consumed per feed. Some publishers expose very long RSS
    # archives (e.g. VietnamNet returns 1000 items/feed); fetching every
    # article at the polite rate blows the per-run time budget. ``None``
    # means no cap and is the historical default.
    max_items_per_feed: int | None = None
    language: str = "vi"


class SourceOut(_Base):
    id: int
    source_id: str
    name: str
    domain: str
    enabled: bool
    robots_ok: bool


# ----------------------------------------------------------------- Article


class ArticleCandidate(_Base):
    """A discovered URL prior to fetching the full article."""

    source_id: str
    url: str
    title: str
    published_at: datetime | None = None
    rss_category: str | None = None
    author: str | None = None


class ArticleCreate(_Base):
    """Validated row about to be inserted into ``articles``."""

    source_fk: int
    url: str
    url_hash: str = Field(min_length=32, max_length=32)
    title: str
    author: str | None = None
    category: str | None = None
    published_at: datetime | None = None
    language: str = "vi"
    raw_html_path: str | None = None
    content_text: str | None = None
    word_count: int = 0
    simhash: int | None = None
    status: ArticleStatus = ArticleStatus.NEW
    error: str | None = None


class ArticleOut(_Base):
    id: int
    source_fk: int
    url: str
    title: str
    author: str | None = None
    category: str | None = None
    published_at: datetime | None = None
    fetched_at: datetime
    language: str
    word_count: int
    status: ArticleStatus


# ----------------------------------------------------------------- Summary


class SummaryOut(_Base):
    id: int
    article_fk: int
    summary_text: str
    model_name: str
    model_version: str | None = None
    prompt_version: str | None = None
    is_published: bool
    created_at: datetime


# ----------------------------------------------------------------- Label


class LabelOut(_Base):
    id: int
    article_fk: int
    source: LabelSource
    summary_text: str
    prompt_version: str | None = None
    quality: LabelQuality | None = None
    qc_passed: bool
    created_at: datetime
