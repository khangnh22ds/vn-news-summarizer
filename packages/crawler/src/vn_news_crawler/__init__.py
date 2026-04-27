"""RSS / sitemap / HTML crawler for Vietnamese news sources."""

from __future__ import annotations

from .config import SourcesConfig, load_sources_config
from .pipeline import CrawlReport, SourceStats, run_once
from .scheduler import make_scheduler

__version__ = "0.1.0"

__all__ = [
    "CrawlReport",
    "SourceStats",
    "SourcesConfig",
    "__version__",
    "load_sources_config",
    "make_scheduler",
    "run_once",
]
