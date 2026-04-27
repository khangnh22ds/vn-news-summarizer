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


class _FakeBatchTokenizer:
    """Returns a dict-like object whose ``.to`` is a no-op (CPU tensors)."""

    pad_token_id = 0

    def __call__(
        self,
        texts: list[str],
        *,
        max_length: int = 1024,
        truncation: bool = True,
        padding: bool = True,
        return_tensors: str = "pt",
    ) -> dict[str, list[list[int]]]:
        del max_length, truncation, padding, return_tensors
        return {
            "input_ids": [[1, 2, 3] for _ in texts],
            "attention_mask": [[1, 1, 1] for _ in texts],
        }

    def batch_decode(
        self, outputs: list[list[int]], *, skip_special_tokens: bool = True
    ) -> list[str]:
        del skip_special_tokens
        return [f"sum:{i}" for i in range(len(outputs))]


class _FakeBatchModel:
    """Records every call to ``generate`` so the test can inspect chunking."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def generate(self, **kwargs: object) -> list[list[int]]:
        ids = kwargs["input_ids"]
        n = len(ids)  # type: ignore[arg-type]
        self.calls.append(n)
        return [[0, 1] for _ in range(n)]

    def to(self, device: str) -> _FakeBatchModel:
        del device
        return self

    def eval(self) -> _FakeBatchModel:
        return self


def test_summarize_batch_chunks_inputs_into_mini_batches(tmp_path: Path) -> None:
    """Beam search would OOM if we passed all 20 articles in one shot; ensure
    summarize_batch breaks them into ``GenerationConfig.batch_size`` chunks."""
    summarizer = ViT5Summarizer(tmp_path, generation=GenerationConfig(batch_size=4))
    fake_model = _FakeBatchModel()
    fake_tokenizer = _FakeBatchTokenizer()
    summarizer._model = fake_model
    summarizer._tokenizer = fake_tokenizer

    out = summarizer.summarize_batch([f"text {i}" for i in range(10)])

    assert len(out) == 10
    # 10 items, batch_size=4 -> chunk sizes [4, 4, 2]
    assert fake_model.calls == [4, 4, 2]


def test_summarize_batch_explicit_batch_size_overrides_generation(tmp_path: Path) -> None:
    summarizer = ViT5Summarizer(tmp_path, generation=GenerationConfig(batch_size=4))
    fake_model = _FakeBatchModel()
    summarizer._model = fake_model
    summarizer._tokenizer = _FakeBatchTokenizer()

    summarizer.summarize_batch(["a", "b", "c", "d", "e"], batch_size=2)
    assert fake_model.calls == [2, 2, 1]


def test_summarize_batch_empty_list_returns_empty(tmp_path: Path) -> None:
    summarizer = ViT5Summarizer(tmp_path)
    # No model load required for an empty list.
    assert summarizer.summarize_batch([]) == []
