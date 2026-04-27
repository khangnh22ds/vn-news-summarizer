"""CLI entrypoint for the evaluation harness (TICKET-004 / Phase 3+).

Examples::

    # Evaluate LexRank on the test split of dataset v1, log to MLflow
    uv run python scripts/run_eval.py --baseline lexrank --dataset v1

    # Skip MLflow + add BERTScore (slow, downloads xlm-roberta-base)
    uv run python scripts/run_eval.py --baseline textrank --dataset v1 \\
        --no-mlflow --bertscore --split val
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import cast

from vn_news_training import (
    BaselineName,
    Example,
    ExtractiveSummarizer,
    SplitName,
    SummarizerConfig,
    evaluate_predictions,
    load_split,
    log_metrics,
    log_params,
    mlflow_run,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_eval",
        description="Evaluate a summarization baseline / fine-tuned model on a dataset version.",
    )
    p.add_argument(
        "--baseline",
        choices=["lexrank", "textrank", "vit5"],
        default="lexrank",
        help=(
            "Which model to run. 'lexrank'/'textrank' are the extractive "
            "baselines; 'vit5' loads the fine-tuned ViT5 / LoRA checkpoint "
            "given by --model-path."
        ),
    )
    p.add_argument(
        "--model-path",
        default=None,
        help=(
            "Path to a ViT5 / LoRA checkpoint. Required when --baseline=vit5. "
            "May also be a base model name like 'VietAI/vit5-base'."
        ),
    )
    p.add_argument(
        "--dataset",
        default="v1",
        help="Dataset version name under data/datasets/ (default: v1).",
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/datasets"),
        help="Root directory holding dataset versions.",
    )
    p.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Which split to evaluate on (default: test).",
    )
    p.add_argument("--max-words", type=int, default=70, help="Hard cap on summary words.")
    p.add_argument("--max-sentences", type=int, default=4, help="Soft cap on sentences.")
    p.add_argument(
        "--bertscore",
        action="store_true",
        help="Also compute BERTScore-F1 (slow, requires xlm-roberta-base).",
    )
    p.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow tracking (still prints metrics to stdout).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit evaluation to first N examples (debug aid).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Stdlib log level for the CLI.",
    )
    return p


def _summarize_all(examples: list[Example], cfg: SummarizerConfig) -> tuple[list[str], list[str]]:
    summarizer = ExtractiveSummarizer(cfg)
    preds: list[str] = []
    refs: list[str] = []
    for ex in examples:
        preds.append(summarizer.summarize(ex.content_text))
        refs.append(ex.summary)
    return preds, refs


def _summarize_all_vit5(examples: list[Example], model_path: str) -> tuple[list[str], list[str]]:
    """Run the fine-tuned ViT5 / LoRA summarizer over a list of examples."""
    from vn_news_inference import ViT5Summarizer

    summarizer = ViT5Summarizer(model_path)
    preds = summarizer.summarize_batch([ex.content_text for ex in examples])
    refs = [ex.summary for ex in examples]
    return preds, refs


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    log = logging.getLogger("run_eval")

    dataset_dir = args.dataset_root / args.dataset
    if not dataset_dir.exists():
        log.error("dataset directory not found: %s", dataset_dir)
        return 2

    split_name = cast(SplitName, args.split)
    examples = load_split(dataset_dir, split_name)
    if args.limit is not None:
        examples = examples[: args.limit]
    log.info("loaded %d examples from %s/%s.jsonl", len(examples), dataset_dir, args.split)
    if not examples:
        log.warning("no examples — nothing to evaluate")
        return 0

    if args.baseline == "vit5":
        if not args.model_path:
            log.error("--model-path is required when --baseline=vit5")
            return 2
        preds, refs = _summarize_all_vit5(examples, args.model_path)
    else:
        cfg = SummarizerConfig(
            name=cast(BaselineName, args.baseline),
            max_words=args.max_words,
            max_sentences=args.max_sentences,
        )
        preds, refs = _summarize_all(examples, cfg)
    result = evaluate_predictions(preds, refs, use_bertscore=args.bertscore)
    metrics = result.to_dict()

    log.info("metrics: %s", json.dumps(metrics, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if not args.no_mlflow:
        with mlflow_run(
            experiment="baselines",
            run_name=f"{args.baseline}-{args.dataset}-{args.split}",
            tags={"baseline": args.baseline, "dataset": args.dataset, "split": args.split},
        ):
            log_params(
                {
                    "baseline": args.baseline,
                    "dataset": args.dataset,
                    "split": args.split,
                    "max_words": args.max_words,
                    "max_sentences": args.max_sentences,
                    "bertscore": args.bertscore,
                }
            )
            log_metrics({k: float(v) for k, v in metrics.items()})

    return 0


if __name__ == "__main__":
    sys.exit(main())
