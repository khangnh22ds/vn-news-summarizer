"""Load + validate the sources YAML config.

The YAML file lives at ``configs/sources.yaml`` and is parsed into a
list of :class:`SourceConfig` (defined in :mod:`vn_news_common.schemas`)
plus a small :class:`Defaults` block.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from vn_news_common.schemas import SourceConfig


class Defaults(BaseModel):
    user_agent: str
    crawl_delay_s: float = 1.0
    timeout_s: float = 20.0
    max_retries: int = 3
    language: str = "vi"


class SourcesConfig(BaseModel):
    defaults: Defaults
    sources: list[SourceConfig]
    canonical_categories: dict[str, list[str]] = Field(default_factory=dict)

    def enabled(self) -> list[SourceConfig]:
        return [s for s in self.sources if s.enabled]


def load_sources_config(path: str | Path) -> SourcesConfig:
    """Parse and validate the sources YAML."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return SourcesConfig.model_validate(raw)


def find_canonical_category(
    raw: str | None,
    *,
    canonical: dict[str, list[str]],
) -> str | None:
    """Map a raw publisher category string to one of our canonical keys.

    Returns the canonical key if any alias matches (case-insensitive
    substring), else ``None``.
    """
    if not raw:
        return None
    needle = raw.lower()
    for key, aliases in canonical.items():
        for alias in aliases:
            if alias.lower() in needle or needle in alias.lower():
                return key
    return None
