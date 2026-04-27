"""Dataset versioning + JSONL export for downstream training.

Picks all QC-passed LLM labels for a given ``prompt_version``, splits
them deterministically into train/val/test (80/10/10 by default), writes
JSONL files under ``data/datasets/<name>/`` and inserts a row into
``dataset_versions``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from sqlalchemy import and_, select
from vn_news_common.db import session_scope
from vn_news_common.enums import LabelSource
from vn_news_common.models import Article, DatasetVersion, Label, Source


@dataclass(slots=True)
class DatasetStats:
    """Result of building a dataset version."""

    name: str
    total: int
    train: int
    val: int
    test: int
    out_dir: Path


def _split_bucket(article_id: int, *, salt: str = "vn-news-v1") -> str:
    """Deterministic 80/10/10 split keyed by article id (stable across runs)."""
    h = hashlib.sha256(f"{salt}:{article_id}".encode()).digest()
    bucket = h[0]
    if bucket < 26:  # ~10%
        return "test"
    if bucket < 52:  # ~10%
        return "val"
    return "train"


async def build_dataset_version(
    *,
    name: str,
    prompt_version: str,
    out_root: Path,
    description: str | None = None,
) -> DatasetStats:
    """Build a new dataset version from QC-passed LLM labels."""
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    train: list[dict[str, object]] = []
    val: list[dict[str, object]] = []
    test: list[dict[str, object]] = []

    async with session_scope() as session:
        stmt = (
            select(Label, Article, Source)
            .join(Article, Article.id == Label.article_fk)
            .join(Source, Source.id == Article.source_fk)
            .where(
                and_(
                    Label.source == LabelSource.LLM,
                    Label.qc_passed.is_(True),
                    Label.prompt_version == prompt_version,
                )
            )
            .order_by(Article.id.asc())
        )
        rows = (await session.execute(stmt)).all()

        train_ids: list[int] = []
        val_ids: list[int] = []
        test_ids: list[int] = []
        for label, article, source in rows:
            row = {
                "article_id": article.id,
                "source": source.source_id,
                "url": article.url,
                "title": article.title,
                "category": article.category,
                "published_at": (
                    article.published_at.isoformat() if article.published_at else None
                ),
                "content_text": article.content_text,
                "summary": label.summary_text,
                "prompt_version": label.prompt_version,
            }
            bucket = _split_bucket(article.id)
            if bucket == "train":
                train.append(row)
                train_ids.append(article.id)
            elif bucket == "val":
                val.append(row)
                val_ids.append(article.id)
            else:
                test.append(row)
                test_ids.append(article.id)

        version_row = DatasetVersion(
            name=name,
            description=description or f"LLM labels at prompt={prompt_version}",
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
        )
        session.add(version_row)

    for split_name, split_rows in (("train", train), ("val", val), ("test", test)):
        path = out_dir / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in split_rows:
                fh.write(json.dumps(r, ensure_ascii=False))
                fh.write("\n")
        logger.info("wrote {} rows to {}", len(split_rows), path)

    return DatasetStats(
        name=name,
        total=len(train) + len(val) + len(test),
        train=len(train),
        val=len(val),
        test=len(test),
        out_dir=out_dir,
    )
