"""Unit tests for the ViT5 / LoRA inference loader (no model download)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from vn_news_inference import GenerationConfig, ViT5Summarizer
from vn_news_inference.finetune_loader import (
    _is_adapter_dir,
    _read_base_model_from_adapter,
)


def test_generation_config_defaults() -> None:
    cfg = GenerationConfig()
    assert cfg.max_input_length == 1024
    assert cfg.num_beams == 4
    assert cfg.early_stopping is True


def test_summarizer_summarize_empty_returns_empty(tmp_path: Path) -> None:
    summarizer = ViT5Summarizer(tmp_path)
    # Empty input must short-circuit *before* trying to load any model.
    assert summarizer.summarize("") == ""
    assert summarizer.summarize("   \n\t") == ""


def test_is_adapter_dir_detects_adapter(tmp_path: Path) -> None:
    assert _is_adapter_dir(tmp_path) is False
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert _is_adapter_dir(tmp_path) is True


def test_read_base_model_from_adapter_happy_path(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "VietAI/vit5-base"}),
        encoding="utf-8",
    )
    assert _read_base_model_from_adapter(tmp_path) == "VietAI/vit5-base"


def test_read_base_model_from_adapter_missing_field_raises(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="base_model_name_or_path"):
        _read_base_model_from_adapter(tmp_path)


def test_summarizer_constructor_does_not_load_model(tmp_path: Path) -> None:
    """The ctor must be cheap — no transformers / torch imports yet."""
    summarizer = ViT5Summarizer(tmp_path, generation=GenerationConfig(num_beams=1))
    assert summarizer._model is None
    assert summarizer._tokenizer is None
    assert summarizer.generation.num_beams == 1
