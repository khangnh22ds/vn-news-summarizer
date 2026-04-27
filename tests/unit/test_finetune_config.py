"""Unit tests for the YAML training config loader."""

from __future__ import annotations

from pathlib import Path

import pytest
from vn_news_training import (
    DatasetConfig,
    FinetuneConfig,
    ModelConfig,
    PeftConfig,
    TrainingConfig,
    load_finetune_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_load_default_vit5_config_round_trips() -> None:
    cfg = load_finetune_config(REPO_ROOT / "configs/training/vit5_base_v1.yaml")
    assert isinstance(cfg, FinetuneConfig)
    assert cfg.run_name == "vit5-base-news-v1"
    assert cfg.model.base == "VietAI/vit5-base"
    assert cfg.model.max_input_length == 1024
    assert cfg.model.max_target_length == 128
    assert cfg.dataset.name == "v1"
    assert cfg.dataset.text_column == "content_text"
    assert cfg.dataset.target_column == "summary"
    assert cfg.training.predict_with_generate is True
    assert cfg.training.metric_for_best_model == "rougeL"
    assert cfg.peft.enabled is True
    assert cfg.peft.r == 16
    assert cfg.peft.target_modules == ["q", "v"]


def test_load_finetune_config_uses_defaults_for_missing_sections(tmp_path: Path) -> None:
    cfg_file = tmp_path / "minimal.yaml"
    cfg_file.write_text(
        """
version: "1.0.0"
run_name: tiny-run
model:
  base: t5-small
  max_input_length: 256
  max_target_length: 32
""",
        encoding="utf-8",
    )
    cfg = load_finetune_config(cfg_file)
    assert cfg.run_name == "tiny-run"
    assert cfg.model.base == "t5-small"
    assert cfg.model.max_input_length == 256
    # Untouched sections fall back to dataclass defaults.
    assert isinstance(cfg.dataset, DatasetConfig)
    assert isinstance(cfg.training, TrainingConfig)
    assert isinstance(cfg.peft, PeftConfig)
    assert cfg.peft.enabled is True


def test_load_finetune_config_ignores_unknown_keys(tmp_path: Path) -> None:
    cfg_file = tmp_path / "extra.yaml"
    cfg_file.write_text(
        """
version: "1.0.0"
model:
  base: t5-small
  ghost_field: should_be_ignored
""",
        encoding="utf-8",
    )
    cfg = load_finetune_config(cfg_file)
    assert cfg.model.base == "t5-small"


def test_load_finetune_config_empty_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_finetune_config("does/not/exist.yaml")


def test_finetune_config_dataclass_defaults() -> None:
    cfg = FinetuneConfig()
    assert isinstance(cfg.model, ModelConfig)
    assert cfg.model.base == "VietAI/vit5-base"
    assert cfg.training.fp16 is True
    assert cfg.training.eval_strategy == "epoch"
    assert cfg.peft.target_modules == ["q", "v"]
