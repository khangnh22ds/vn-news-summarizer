"""Enum types shared across ORM, schemas, and code."""

from __future__ import annotations

from enum import StrEnum


class ArticleStatus(StrEnum):
    """Lifecycle status of an article row."""

    NEW = "new"
    CLEANED = "cleaned"
    SUMMARIZED = "summarized"
    FAILED = "failed"


class LabelSource(StrEnum):
    """Origin of a label / summary used for training data."""

    LLM = "llm"
    HUMAN = "human"
    LLM_HUMAN = "llm+human"


class LabelQuality(StrEnum):
    """Reviewer disposition for a label."""

    GOOD = "good"
    BAD = "bad"
    FIX = "fix"


class CrawlRunStatus(StrEnum):
    """Final state of a crawl run."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
