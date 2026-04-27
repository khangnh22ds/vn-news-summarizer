"""CLI entrypoint for the LLM labeling worker (TICKET-003 / Phase 2).

Examples
--------
Label 50 unlabeled articles with the default prompt::

    uv run python scripts/run_labeler.py --limit 50

Label only articles from VnExpress and TuoiTre::

    uv run python scripts/run_labeler.py --source vnexpress --source tuoitre

Build a JSONL dataset from all QC-passed labels::

    uv run python scripts/run_labeler.py --export v1 --no-label
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger
from vn_news_common.db import dispose_engine
from vn_news_common.settings import get_settings
from vn_news_labeling import (
    GenerationParams,
    Prompt,
    VertexLabeler,
    build_dataset_version,
    label_batch,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = REPO_ROOT / "configs" / "prompts" / "summarize_v1.yaml"
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "datasets"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM labeling worker (TICKET-003 / Phase 2).")
    p.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    p.add_argument("--limit", type=int, default=50, help="Max articles to label this run.")
    p.add_argument(
        "--source",
        action="append",
        default=None,
        help="Limit to specific source ids (repeatable).",
    )
    p.add_argument(
        "--no-label",
        action="store_true",
        help="Skip the LLM labeling pass — useful when only --export is desired.",
    )
    p.add_argument(
        "--export",
        type=str,
        default=None,
        help="If set, build a dataset version with the given name (e.g. 'v1').",
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Output root for JSONL files.",
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return p


def _configure_logging(level: str) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level)


def _build_labeler(prompt: Prompt) -> VertexLabeler:
    settings = get_settings()
    params = GenerationParams(
        temperature=prompt.generation.temperature,
        top_p=prompt.generation.top_p,
        max_output_tokens=prompt.generation.max_output_tokens,
        response_mime_type=prompt.generation.response_mime_type,
    )
    return VertexLabeler(
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
        model_name=prompt.model or settings.vertex_llm_model,
        params=params,
    )


async def _run(args: argparse.Namespace) -> int:
    prompt = Prompt.from_yaml(args.prompt)
    logger.info(
        "loaded prompt v{} model={} provider={}",
        prompt.version,
        prompt.model,
        prompt.provider,
    )

    if not args.no_label:
        labeler = _build_labeler(prompt)
        stats = await label_batch(
            prompt=prompt,
            labeler=labeler,
            limit=args.limit,
            only_sources=args.source,
        )
        logger.info(
            "labeling done: requested={} labeled={} qc_pass={} qc_fail={} llm_err={} (rate={:.1%})",
            stats.requested,
            stats.labeled,
            stats.qc_passed,
            stats.qc_failed,
            stats.llm_errors,
            stats.qc_pass_rate,
        )

    if args.export:
        ds = await build_dataset_version(
            name=args.export,
            prompt_version=prompt.version,
            out_root=args.dataset_root,
        )
        logger.info(
            "exported dataset {} ({} rows: train={} val={} test={}) -> {}",
            ds.name,
            ds.total,
            ds.train,
            ds.val,
            ds.test,
            ds.out_dir,
        )

    await dispose_engine()
    return 0


def main() -> int:
    args = _build_arg_parser().parse_args()
    _configure_logging(args.log_level)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
