"""SQLAlchemy 2 ORM models.

Schema mirrors ``docs/architecture.md``. All datetimes are stored
timezone-aware (UTC); SQLite renders these as ISO-8601 strings, Postgres
as ``TIMESTAMPTZ``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    MappedAsDataclass,
    mapped_column,
    relationship,
)

from vn_news_common.enums import (
    ArticleStatus,
    CrawlRunStatus,
    LabelQuality,
    LabelSource,
)
from vn_news_common.time_utils import utcnow


class Base(MappedAsDataclass, DeclarativeBase):
    """Base class for all ORM models. Uses dataclass semantics for clarity."""


# --------------------------------------------------------------------- helpers


def _tz_dt() -> Any:
    """Helper for column type used everywhere."""
    return DateTime(timezone=True)


# ----------------------------------------------------------------------- Source


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, init=False)
    source_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128))
    domain: Mapped[str] = mapped_column(String(256))
    rss_urls: Mapped[list[str]] = mapped_column(JSON, default_factory=list)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    robots_ok: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(_tz_dt(), default_factory=utcnow)
    updated_at: Mapped[datetime] = mapped_column(_tz_dt(), default_factory=utcnow)

    articles: Mapped[list[Article]] = relationship(
        back_populates="source",
        cascade="all, delete-orphan",
        default_factory=list,
        repr=False,
    )
    crawl_runs: Mapped[list[CrawlRun]] = relationship(
        back_populates="source",
        cascade="all, delete-orphan",
        default_factory=list,
        repr=False,
    )


# ---------------------------------------------------------------------- Article


class Article(Base):
    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, init=False)
    source_fk: Mapped[int] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"),
        index=True,
    )
    url: Mapped[str] = mapped_column(String(2048), unique=True)
    url_hash: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(1024))

    author: Mapped[str | None] = mapped_column(String(256), default=None)
    category: Mapped[str | None] = mapped_column(String(128), default=None)
    published_at: Mapped[datetime | None] = mapped_column(_tz_dt(), default=None)
    fetched_at: Mapped[datetime] = mapped_column(_tz_dt(), default_factory=utcnow)
    language: Mapped[str] = mapped_column(String(8), default="vi")
    raw_html_path: Mapped[str | None] = mapped_column(String(512), default=None)
    content_text: Mapped[str | None] = mapped_column(Text, default=None)
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    simhash: Mapped[int | None] = mapped_column(BigInteger, default=None, index=True)
    status: Mapped[ArticleStatus] = mapped_column(String(16), default=ArticleStatus.NEW, index=True)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(_tz_dt(), default_factory=utcnow)
    updated_at: Mapped[datetime] = mapped_column(_tz_dt(), default_factory=utcnow)

    source: Mapped[Source] = relationship(back_populates="articles", default=None, repr=False)
    summaries: Mapped[list[Summary]] = relationship(
        back_populates="article",
        cascade="all, delete-orphan",
        default_factory=list,
        repr=False,
    )
    labels: Mapped[list[Label]] = relationship(
        back_populates="article",
        cascade="all, delete-orphan",
        default_factory=list,
        repr=False,
    )

    __table_args__ = (
        Index("ix_articles_published_at_desc", "published_at"),
        Index("ix_articles_status_pub", "status", "published_at"),
    )


# ---------------------------------------------------------------------- Summary


class Summary(Base):
    __tablename__ = "summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, init=False)
    article_fk: Mapped[int] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"), index=True
    )
    summary_text: Mapped[str] = mapped_column(Text)
    model_name: Mapped[str] = mapped_column(String(128))
    model_version: Mapped[str | None] = mapped_column(String(64), default=None)
    prompt_version: Mapped[str | None] = mapped_column(String(32), default=None)

    rouge1: Mapped[float | None] = mapped_column(default=None)
    rouge2: Mapped[float | None] = mapped_column(default=None)
    rougeL: Mapped[float | None] = mapped_column(default=None)
    bertscore_f1: Mapped[float | None] = mapped_column(default=None)
    factuality_score: Mapped[float | None] = mapped_column(default=None)

    is_published: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(_tz_dt(), default_factory=utcnow)

    article: Mapped[Article] = relationship(back_populates="summaries", default=None, repr=False)

    __table_args__ = (Index("ix_summaries_article_pub", "article_fk", "is_published"),)


# ------------------------------------------------------------------------ Label


class Label(Base):
    __tablename__ = "labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, init=False)
    article_fk: Mapped[int] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"), index=True
    )
    source: Mapped[LabelSource] = mapped_column(String(16))
    summary_text: Mapped[str] = mapped_column(Text)
    prompt_version: Mapped[str | None] = mapped_column(String(32), default=None)
    reviewer_id: Mapped[str | None] = mapped_column(String(64), default=None)
    quality: Mapped[LabelQuality | None] = mapped_column(String(16), default=None)
    notes: Mapped[str | None] = mapped_column(Text, default=None)
    qc_passed: Mapped[bool] = mapped_column(Boolean, default=False)
    qc_details: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(_tz_dt(), default_factory=utcnow)

    article: Mapped[Article] = relationship(back_populates="labels", default=None, repr=False)


# ------------------------------------------------------------------- Dataset / Run


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, init=False)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    description: Mapped[str | None] = mapped_column(Text, default=None)
    train_ids: Mapped[list[int]] = mapped_column(JSON, default_factory=list)
    val_ids: Mapped[list[int]] = mapped_column(JSON, default_factory=list)
    test_ids: Mapped[list[int]] = mapped_column(JSON, default_factory=list)
    created_at: Mapped[datetime] = mapped_column(_tz_dt(), default_factory=utcnow)


class CrawlRun(Base):
    __tablename__ = "crawl_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, init=False)
    source_fk: Mapped[int] = mapped_column(ForeignKey("sources.id", ondelete="CASCADE"), index=True)
    started_at: Mapped[datetime] = mapped_column(_tz_dt(), default_factory=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(_tz_dt(), default=None)
    n_new: Mapped[int] = mapped_column(Integer, default=0)
    n_skipped: Mapped[int] = mapped_column(Integer, default=0)
    n_errors: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[CrawlRunStatus] = mapped_column(String(16), default=CrawlRunStatus.SUCCESS)
    notes: Mapped[str | None] = mapped_column(Text, default=None)

    source: Mapped[Source] = relationship(back_populates="crawl_runs", default=None, repr=False)


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, init=False)
    model_name: Mapped[str] = mapped_column(String(128))
    version: Mapped[str] = mapped_column(String(64))
    params: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    artifact_path: Mapped[str | None] = mapped_column(String(512), default=None)
    dataset_version_fk: Mapped[int | None] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="SET NULL"), default=None
    )
    created_at: Mapped[datetime] = mapped_column(_tz_dt(), default_factory=utcnow)

    __table_args__ = (UniqueConstraint("model_name", "version", name="uq_model_runs_name_version"),)
