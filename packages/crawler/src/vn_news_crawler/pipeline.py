"""End-to-end crawl pipeline.

Steps per source:

1. **discover**  RSS → :class:`ArticleCandidate` list
2. **filter**    skip URLs already in DB (``url_hash``) and robots-disallowed
3. **fetch**     polite GET of the article HTML
4. **extract**   trafilatura + readability fallback
5. **dedupe**    SimHash near-duplicate within recent window
6. **persist**   insert :class:`Article` row (status=cleaned)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from vn_news_common.db import session_scope
from vn_news_common.enums import ArticleStatus, CrawlRunStatus
from vn_news_common.models import Article, CrawlRun, Source
from vn_news_common.schemas import ArticleCandidate, SourceConfig
from vn_news_common.text import normalize_text, simhash64
from vn_news_common.time_utils import utcnow
from vn_news_common.url_utils import canonicalize_url, url_hash

from .config import SourcesConfig, find_canonical_category
from .dedupe import find_near_duplicate
from .extract import ExtractedArticle, extract_from_html
from .http_client import PoliteClient
from .robots import RobotsCache
from .rss import fetch_feed


@dataclass(slots=True)
class SourceStats:
    source_id: str
    discovered: int = 0
    skipped_seen: int = 0
    skipped_robots: int = 0
    skipped_dupe: int = 0
    fetch_failed: int = 0
    extract_failed: int = 0
    inserted: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CrawlReport:
    started_at: datetime
    ended_at: datetime
    per_source: list[SourceStats] = field(default_factory=list)

    @property
    def total_inserted(self) -> int:
        return sum(s.inserted for s in self.per_source)


# ------------------------------------------------------------------ helpers


async def _ensure_source(session: AsyncSession, src: SourceConfig) -> Source:
    """Return the DB row for a source, creating/updating it as needed."""
    stmt = select(Source).where(Source.source_id == src.id)
    existing: Source | None = (await session.execute(stmt)).scalar_one_or_none()
    rss_urls = [str(u) for u in src.rss]
    if existing is None:
        row = Source(
            source_id=src.id,
            name=src.name,
            domain=src.domain,
            rss_urls=rss_urls,
            enabled=src.enabled,
        )
        session.add(row)
        await session.flush()
        return row
    existing.name = src.name
    existing.domain = src.domain
    existing.rss_urls = rss_urls
    existing.enabled = src.enabled
    existing.updated_at = utcnow()
    await session.flush()
    return existing


async def _existing_url_hashes(session: AsyncSession, hashes: list[str]) -> set[str]:
    if not hashes:
        return set()
    stmt = select(Article.url_hash).where(Article.url_hash.in_(hashes))
    res = await session.execute(stmt)
    return {row[0] for row in res.all()}


# ---------------------------------------------------------------- per source


async def _crawl_one_source(
    *,
    src: SourceConfig,
    cfg: SourcesConfig,
    http: PoliteClient,
    robots: RobotsCache,
) -> SourceStats:
    stats = SourceStats(source_id=src.id)

    if not src.rss:
        stats.errors.append("no rss feeds configured")
        return stats

    candidates: list[ArticleCandidate] = []
    seen_urls: set[str] = set()
    for feed_url in src.rss:
        feed_items = await fetch_feed(http, source_id=src.id, feed_url=str(feed_url))
        for c in feed_items:
            canon = canonicalize_url(c.url)
            if canon in seen_urls:
                continue
            seen_urls.add(canon)
            candidates.append(c)

    stats.discovered = len(candidates)

    async with session_scope() as session:
        source_row = await _ensure_source(session, src)

        # L1 dedup: skip URLs already stored.
        all_hashes = [url_hash(c.url) for c in candidates]
        already = await _existing_url_hashes(session, all_hashes)

        keep: list[tuple[ArticleCandidate, str]] = []
        for c, h in zip(candidates, all_hashes, strict=True):
            if h in already:
                stats.skipped_seen += 1
                continue
            allowed = await robots.can_fetch(c.url)
            if not allowed:
                stats.skipped_robots += 1
                continue
            keep.append((c, h))

        for c, h in keep:
            try:
                res = await http.get(c.url)
            except Exception as exc:
                logger.warning("fetch failed {}: {}", c.url, exc)
                stats.fetch_failed += 1
                continue
            extracted = extract_from_html(res.text, url=c.url)
            if extracted is None:
                stats.extract_failed += 1
                continue

            text = extracted.content_text
            simhash_val = simhash64(text)
            dup_id = await find_near_duplicate(session, text=text)
            if dup_id is not None:
                stats.skipped_dupe += 1
                continue

            article = _build_article_row(
                source_row=source_row,
                candidate=c,
                extracted=extracted,
                url_hash_val=h,
                simhash_val=simhash_val,
                cfg=cfg,
            )
            session.add(article)
            try:
                await session.flush()
                stats.inserted += 1
            except IntegrityError:
                logger.warning("integrity skip {}", c.url)
                await session.rollback()
                stats.skipped_seen += 1

    return stats


def _build_article_row(
    *,
    source_row: Source,
    candidate: ArticleCandidate,
    extracted: ExtractedArticle,
    url_hash_val: str,
    simhash_val: int,
    cfg: SourcesConfig,
) -> Article:
    raw_category = candidate.rss_category
    canon_category = find_canonical_category(raw_category, canonical=cfg.canonical_categories)
    article = Article(
        source_fk=source_row.id,
        url=canonicalize_url(candidate.url),
        url_hash=url_hash_val,
        title=normalize_text(extracted.title or candidate.title),
        author=extracted.author or candidate.author,
        category=canon_category or raw_category,
        published_at=extracted.published_at or candidate.published_at,
        language=extracted.language or "vi",
        content_text=extracted.content_text,
        word_count=extracted.word_count,
        simhash=simhash_val,
        status=ArticleStatus.CLEANED,
    )
    # Attach the relationship so SQLAlchemy doesn't null out
    # ``source_fk`` via the ``back_populates="articles"`` linkage on flush.
    article.source = source_row
    return article


# -------------------------------------------------------------------- driver


async def run_once(
    cfg: SourcesConfig,
    *,
    only_sources: list[str] | None = None,
) -> CrawlReport:
    """Run one full crawl pass over all enabled sources."""
    started = utcnow()
    report = CrawlReport(started_at=started, ended_at=started)
    enabled = [s for s in cfg.enabled() if not only_sources or s.id in only_sources]
    if not enabled:
        report.ended_at = utcnow()
        return report

    user_agent = cfg.defaults.user_agent
    http = PoliteClient(
        user_agent=user_agent,
        timeout_s=cfg.defaults.timeout_s,
        rps_per_host=1.0 / cfg.defaults.crawl_delay_s if cfg.defaults.crawl_delay_s else 1.0,
        max_retries=cfg.defaults.max_retries,
    )
    robots = RobotsCache(user_agent=user_agent)

    try:
        for src in enabled:
            run_started = utcnow()
            try:
                stats = await _crawl_one_source(src=src, cfg=cfg, http=http, robots=robots)
                logger.info(
                    "[{}] discovered={} new={} dupe={} robots={} fetch_fail={} extract_fail={}",
                    src.id,
                    stats.discovered,
                    stats.inserted,
                    stats.skipped_dupe,
                    stats.skipped_robots,
                    stats.fetch_failed,
                    stats.extract_failed,
                )
            except Exception as exc:
                logger.exception("source {} crashed: {}", src.id, exc)
                stats = SourceStats(source_id=src.id, errors=[str(exc)])
            report.per_source.append(stats)
            await _record_crawl_run(src=src, stats=stats, started_at=run_started)
    finally:
        await asyncio.gather(http.aclose(), robots.aclose(), return_exceptions=True)

    report.ended_at = utcnow()
    return report


async def _record_crawl_run(
    *,
    src: SourceConfig,
    stats: SourceStats,
    started_at: datetime,
) -> None:
    """Persist a crawl_runs row for observability."""
    async with session_scope() as session:
        source_row = (
            await session.execute(select(Source).where(Source.source_id == src.id))
        ).scalar_one()
        status = (
            CrawlRunStatus.SUCCESS
            if stats.inserted > 0 and not stats.errors
            else CrawlRunStatus.PARTIAL
            if stats.inserted > 0
            else CrawlRunStatus.FAILED
            if stats.errors and not stats.discovered
            else CrawlRunStatus.PARTIAL
        )
        notes = (
            f"discovered={stats.discovered} inserted={stats.inserted} "
            f"dupe={stats.skipped_dupe} seen={stats.skipped_seen} "
            f"robots={stats.skipped_robots} fetch_fail={stats.fetch_failed} "
            f"extract_fail={stats.extract_failed}"
            + (f" errors={'; '.join(stats.errors)}" if stats.errors else "")
        )
        run = CrawlRun(
            source_fk=source_row.id,
            started_at=started_at,
            ended_at=utcnow(),
            n_new=stats.inserted,
            n_skipped=stats.skipped_seen + stats.skipped_dupe + stats.skipped_robots,
            n_errors=stats.fetch_failed + stats.extract_failed + len(stats.errors),
            status=status,
            notes=notes,
        )
        run.source = source_row
        session.add(run)
