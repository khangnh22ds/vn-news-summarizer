"""Smoke tests for the ``scripts/run_training.py`` CLI."""

from __future__ import annotations

from pathlib import Path

import run_training  # type: ignore[import-not-found]


def test_help_text_mentions_finetune() -> None:
    parser = run_training._build_parser()
    help_text = parser.format_help()
    assert "Fine-tune ViT5" in help_text
    assert "--config" in help_text
    assert "--epochs" in help_text


def test_main_returns_2_when_config_missing(tmp_path: Path) -> None:
    rc = run_training.main(["--config", str(tmp_path / "missing.yaml")])
    assert rc == 2
