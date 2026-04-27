"""CLI entrypoint for the crawler.

Usage::

    # one-shot crawl over all enabled sources
    uv run python scripts/run_crawler.py

    # crawl only specific sources (id matches configs/sources.yaml)
    uv run python scripts/run_crawler.py --source vnexpress --source tuoitre

    # run as a long-lived scheduler (every 30 minutes by default)
    uv run python scripts/run_crawler.py --schedule

    # custom interval (in minutes) for the scheduler
    uv run python scripts/run_crawler.py --schedule --interval 60
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "sources.yaml"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="vn-news crawler")
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to sources YAML (default: configs/sources.yaml).",
    )
    p.add_argument(
        "--source",
        action="append",
        default=None,
        help="Restrict to a specific source id; repeatable.",
    )
    p.add_argument(
        "--schedule",
        action="store_true",
        help="Run as a long-lived APScheduler instead of a one-shot.",
    )
    p.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Scheduler interval in minutes (default: 30).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


async def _run_once(args: argparse.Namespace) -> int:
    from vn_news_crawler import load_sources_config, run_once

    cfg = load_sources_config(args.config)
    report = await run_once(cfg, only_sources=args.source)
    logger.info(
        "crawl done: total_inserted={} sources={}",
        report.total_inserted,
        len(report.per_source),
    )
    for s in report.per_source:
        logger.info(
            "  {}: discovered={} new={} dupe={} robots={} fetch_fail={} extract_fail={}",
            s.source_id,
            s.discovered,
            s.inserted,
            s.skipped_dupe,
            s.skipped_robots,
            s.fetch_failed,
            s.extract_failed,
        )
    return 0 if report.total_inserted > 0 else 1


async def _run_schedule(args: argparse.Namespace) -> int:
    from vn_news_crawler import load_sources_config, make_scheduler, run_once

    cfg = load_sources_config(args.config)
    sched = make_scheduler(cfg, interval_minutes=args.interval, only_sources=args.source)
    sched.start()
    logger.info("scheduler started (interval={}min) — Ctrl+C to stop", args.interval)
    # Run once immediately, then block forever.
    await run_once(cfg, only_sources=args.source)
    stop_event = asyncio.Event()
    try:
        await stop_event.wait()
    except asyncio.CancelledError:  # pragma: no cover
        pass
    finally:
        sched.shutdown(wait=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    if not args.config.exists():
        logger.error("config not found: {}", args.config)
        return 2

    runner = _run_schedule if args.schedule else _run_once
    return asyncio.run(runner(args))


if __name__ == "__main__":
    raise SystemExit(main())
