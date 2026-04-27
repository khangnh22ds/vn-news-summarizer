"""Extractive baselines for Phase 3 (TICKET-004)."""

from __future__ import annotations

from .extractive import (
    BaselineName,
    ExtractiveSummarizer,
    SummarizerConfig,
)
from .tokenizer import sent_tokenize, word_tokenize

__all__ = [
    "BaselineName",
    "ExtractiveSummarizer",
    "SummarizerConfig",
    "sent_tokenize",
    "word_tokenize",
]
