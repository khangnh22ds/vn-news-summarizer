"""JSONL dataset loader for the train/val/test splits exported by the
labeling pipeline (``packages/labeling.dataset.build_dataset_version``).

The exported records have shape::

    {
        "article_id": int,
        "source": str,
        "url": str,
        "title": str,
        "category": str | None,
        "published_at": str | None,
        "content_text": str,
        "summary": str,
        "prompt_version": str,
    }

We expose a small typed wrapper that downstream baselines / training code
can iterate over without needing to know the on-disk format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SplitName = Literal["train", "val", "test"]


@dataclass(slots=True, frozen=True)
class Example:
    """A single labeled article, as written by the labeler."""

    article_id: int
    source: str
    url: str
    title: str
    category: str | None
    published_at: str | None
    content_text: str
    summary: str
    prompt_version: str


def _example_from_dict(raw: dict[str, object]) -> Example:
    aid = raw.get("article_id")
    article_id = int(aid) if isinstance(aid, int | str) else 0
    return Example(
        article_id=article_id,
        source=str(raw.get("source") or ""),
        url=str(raw.get("url") or ""),
        title=str(raw.get("title") or ""),
        category=(str(raw["category"]) if raw.get("category") is not None else None),
        published_at=(str(raw["published_at"]) if raw.get("published_at") is not None else None),
        content_text=str(raw.get("content_text") or ""),
        summary=str(raw.get("summary") or ""),
        prompt_version=str(raw.get("prompt_version") or ""),
    )


def load_split(dataset_dir: Path | str, split: SplitName) -> list[Example]:
    """Load ``<dataset_dir>/<split>.jsonl`` into a list of :class:`Example`.

    Empty files return ``[]``. Missing files raise :class:`FileNotFoundError`.
    """
    path = Path(dataset_dir) / f"{split}.jsonl"
    if not path.exists():
        msg = f"split file not found: {path}"
        raise FileNotFoundError(msg)
    out: list[Example] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            out.append(_example_from_dict(json.loads(line)))
    return out


def load_dataset(
    dataset_dir: Path | str,
) -> dict[SplitName, list[Example]]:
    """Load all three splits at once."""
    return {
        "train": load_split(dataset_dir, "train"),
        "val": load_split(dataset_dir, "val"),
        "test": load_split(dataset_dir, "test"),
    }
