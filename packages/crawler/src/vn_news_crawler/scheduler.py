"""APScheduler wiring for the periodic crawler.

The schedule is managed by an :class:`AsyncIOScheduler` running inside
the asyncio event loop. We use a single ``IntervalTrigger`` (default 30
minutes) and ``coalesce=True`` so missed runs don't pile up.
"""

from __future__ import annotations

from datetime import UTC, datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from .config import SourcesConfig
from .pipeline import run_once


def make_scheduler(
    cfg: SourcesConfig,
    *,
    interval_minutes: int = 30,
    only_sources: list[str] | None = None,
) -> AsyncIOScheduler:
    """Build an :class:`AsyncIOScheduler` configured to crawl periodically."""
    sched = AsyncIOScheduler()

    async def _job() -> None:
        logger.info("crawl tick (interval={}min)", interval_minutes)
        report = await run_once(cfg, only_sources=only_sources)
        logger.info("crawl done: total inserted={}", report.total_inserted)

    sched.add_job(
        _job,
        trigger=IntervalTrigger(minutes=interval_minutes),
        id="periodic_crawl",
        max_instances=1,
        coalesce=True,
        # Run once immediately on startup, then every ``interval_minutes``.
        # Passing ``next_run_time=None`` would create a *paused* job; omitting
        # it would defer the first run by ``interval_minutes``.
        next_run_time=datetime.now(UTC),
    )
    return sched
