"""CLI entrypoint for fine-tuning ViT5 (TICKET-005 / Phase 4).

Designed to be invoked from a Colab/Kaggle T4 notebook (heavy training)
or locally if a GPU is available.

Examples::

    # Default config (configs/training/vit5_base_v1.yaml)
    uv run python scripts/run_training.py

    # Custom config
    uv run python scripts/run_training.py --config configs/training/my.yaml

    # Override the dataset version + output dir from the CLI
    uv run python scripts/run_training.py \\
        --dataset v2 --output-dir models/vit5-news-v2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from vn_news_training import load_finetune_config
from vn_news_training.finetune import run_finetune


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_training",
        description="Fine-tune ViT5-base + LoRA on a labeled JSONL dataset.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/vit5_base_v1.yaml"),
        help="Path to the training YAML config.",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="Override dataset.name from the config (e.g. v1).",
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override dataset.root from the config.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override training.output_dir from the config.",
    )
    p.add_argument(
        "--epochs",
        type=float,
        default=None,
        help="Override training.num_train_epochs.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    log = logging.getLogger("run_training")

    if not args.config.exists():
        log.error("config not found: %s", args.config)
        return 2

    cfg = load_finetune_config(args.config)
    if args.dataset:
        cfg.dataset.name = args.dataset
    if args.dataset_root is not None:
        cfg.dataset.root = str(args.dataset_root)
    if args.output_dir is not None:
        cfg.training.output_dir = str(args.output_dir)
    if args.epochs is not None:
        cfg.training.num_train_epochs = args.epochs

    log.info(
        "fine-tuning %s on dataset=%s/%s -> %s (epochs=%s)",
        cfg.model.base,
        cfg.dataset.root,
        cfg.dataset.name,
        cfg.training.output_dir,
        cfg.training.num_train_epochs,
    )

    result = run_finetune(args.config, cfg=cfg)
    log.info(
        "done; checkpoint=%s metrics=%s",
        result.best_model_checkpoint,
        json.dumps(result.metrics, ensure_ascii=False),
    )
    print(json.dumps(result.metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
