"""Orchestrator for one labeling pass.

Reads ``CLEANED`` articles from the DB, runs them through the LLM,
applies QC, and persists the result as a :class:`Label` row (source =
``LabelSource.LLM``). Articles whose label passes QC are also marked
``ArticleStatus.SUMMARIZED`` so downstream tickets (training/inference)
can pick them up.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger
from sqlalchemy import and_, exists, select
from sqlalchemy.exc import SQLAlchemyError
from vn_news_common.db import session_scope
from vn_news_common.enums import ArticleStatus, LabelQuality, LabelSource
from vn_news_common.models import Article, Label, Source
from vn_news_common.time_utils import utcnow

from .prompt import LabelOutput, Prompt, parse_label_json
from .qc import QcResult, run_qc
from .vertex_client import VertexLabeler, VertexLLMError, VertexTransientError


@dataclass(slots=True)
class LabelStats:
    """Per-batch labeling stats."""

    requested: int = 0
    labeled: int = 0
    qc_passed: int = 0
    qc_failed: int = 0
    llm_errors: int = 0
    started_at: datetime = field(default_factory=utcnow)
    ended_at: datetime = field(default_factory=utcnow)

    @property
    def qc_pass_rate(self) -> float:
        return self.qc_passed / self.labeled if self.labeled else 0.0


async def _select_unlabeled_articles(
    *,
    limit: int,
    only_sources: Sequence[str] | None,
    prompt_version: str,
) -> list[Article]:
    """Fetch up to ``limit`` CLEANED articles that lack an LLM label for this prompt version."""
    async with session_scope() as session:
        already_labeled = (
            select(Label.article_fk)
            .where(
                and_(
                    Label.source == LabelSource.LLM,
                    Label.prompt_version == prompt_version,
                )
            )
            .scalar_subquery()
        )
        stmt = (
            select(Article)
            .where(
                Article.status.in_([ArticleStatus.CLEANED, ArticleStatus.SUMMARIZED]),
                Article.content_text.is_not(None),
                Article.id.not_in(already_labeled),
            )
            .order_by(Article.published_at.desc().nulls_last(), Article.id.desc())
            .limit(limit)
        )
        if only_sources:
            stmt = stmt.where(
                exists().where(
                    and_(Source.id == Article.source_fk, Source.source_id.in_(only_sources))
                )
            )
        rows = (await session.execute(stmt)).scalars().all()
        # Detach from session so we can use them after context exits.
        for r in rows:
            session.expunge(r)
        return list(rows)


async def _persist_label(
    *,
    article_id: int,
    output: LabelOutput,
    qc_result: QcResult,
    prompt_version: str,
) -> None:
    """Insert a Label row + bump article status if QC passed."""
    async with session_scope() as session:
        # Load the parent first so we can attach the relationship explicitly
        # — same workaround we use for Article.source: ``MappedAsDataclass`` +
        # ``back_populates`` will otherwise null out the FK on flush.
        article = (
            await session.execute(select(Article).where(Article.id == article_id))
        ).scalar_one()

        label = Label(
            article_fk=article.id,
            source=LabelSource.LLM,
            summary_text=output.summary.strip(),
            prompt_version=prompt_version,
            reviewer_id=None,
            quality=LabelQuality.GOOD if qc_result.passed else None,
            notes=None,
            qc_passed=qc_result.passed,
            qc_details=qc_result.to_dict(),
        )
        label.article = article
        session.add(label)

        if qc_result.passed and article.status != ArticleStatus.SUMMARIZED:
            article.status = ArticleStatus.SUMMARIZED
            article.updated_at = utcnow()


async def label_batch(
    *,
    prompt: Prompt,
    labeler: VertexLabeler,
    limit: int = 50,
    only_sources: Sequence[str] | None = None,
    source_name_lookup: dict[int, str] | None = None,
) -> LabelStats:
    """Label up to ``limit`` articles. Returns :class:`LabelStats`."""
    stats = LabelStats(started_at=utcnow())

    articles = await _select_unlabeled_articles(
        limit=limit,
        only_sources=list(only_sources) if only_sources else None,
        prompt_version=prompt.version,
    )
    stats.requested = len(articles)
    if not articles:
        stats.ended_at = utcnow()
        return stats

    # Build a source-name lookup once if the caller didn't provide one.
    if source_name_lookup is None:
        async with session_scope() as session:
            rows = (await session.execute(select(Source.id, Source.name))).all()
            source_name_lookup = {sid: name for sid, name in rows}

    for art in articles:
        source_name = source_name_lookup.get(art.source_fk, "unknown")
        user_msg = prompt.render_user(
            title=art.title,
            category=art.category,
            source=source_name,
            content_text=art.content_text or "",
        )
        try:
            raw = labeler.generate(system=prompt.system, user=user_msg)
            output = parse_label_json(raw)
        except (VertexLLMError, VertexTransientError, ValueError) as exc:
            # ``VertexTransientError`` is re-raised by tenacity after the
            # 5-attempt retry budget is exhausted; we must not let it bubble
            # up and abort processing of the remaining articles.
            logger.warning("LLM error on article {}: {}", art.id, exc)
            stats.llm_errors += 1
            continue

        qc_result = run_qc(output=output, source_text=art.content_text or "", cfg=prompt.qc)
        try:
            await _persist_label(
                article_id=art.id,
                output=output,
                qc_result=qc_result,
                prompt_version=prompt.version,
            )
        except SQLAlchemyError as exc:  # pragma: no cover — defensive
            logger.exception("DB error persisting label for article {}: {}", art.id, exc)
            stats.llm_errors += 1
            continue

        stats.labeled += 1
        if qc_result.passed:
            stats.qc_passed += 1
        else:
            stats.qc_failed += 1
        logger.info(
            "labeled article {} qc={} reasons={}",
            art.id,
            "PASS" if qc_result.passed else "FAIL",
            ",".join(qc_result.reasons) if qc_result.reasons else "-",
        )

    stats.ended_at = utcnow()
    return stats
