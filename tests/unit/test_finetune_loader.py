"""Unit tests for the ViT5 / LoRA inference loader (no model download)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from vn_news_inference import DEFAULT_HF_REPO, GenerationConfig, ViT5Summarizer
from vn_news_inference.finetune_loader import (
    _is_adapter_dir,
    _read_base_model_from_adapter,
    _try_fetch_hub_adapter_config,
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


def test_summarize_batch_short_circuits_empty_strings(tmp_path: Path) -> None:
    """Empty / whitespace inputs must map to "" without ever hitting the
    model — same contract as :meth:`ViT5Summarizer.summarize`."""
    summarizer = ViT5Summarizer(tmp_path, generation=GenerationConfig(batch_size=2))
    fake_model = _FakeBatchModel()
    summarizer._model = fake_model
    summarizer._tokenizer = _FakeBatchTokenizer()

    out = summarizer.summarize_batch(["hello", "", "  \n", "world"])
    assert out[0] != ""
    assert out[1] == ""
    assert out[2] == ""
    assert out[3] != ""
    # Only the two non-empty inputs reach generate(); batch_size=2 -> [2].
    assert fake_model.calls == [2]


def test_summarize_batch_all_empty_skips_model(tmp_path: Path) -> None:
    summarizer = ViT5Summarizer(tmp_path)
    fake_model = _FakeBatchModel()
    summarizer._model = fake_model
    summarizer._tokenizer = _FakeBatchTokenizer()

    out = summarizer.summarize_batch(["", "  ", "\t\n"])
    assert out == ["", "", ""]
    # Model never called.
    assert fake_model.calls == []


def test_default_hf_repo_is_a_namespaced_id() -> None:
    """The constant must be a ``namespace/name`` Hub id, not a local path."""
    assert "/" in DEFAULT_HF_REPO
    assert not Path(DEFAULT_HF_REPO).exists()


def _install_fake_huggingface_hub(
    monkeypatch: pytest.MonkeyPatch,
    download_impl: object,
) -> None:
    """Replace ``huggingface_hub`` in ``sys.modules`` with a stub so that
    ``importlib.import_module("huggingface_hub")`` inside the loader
    returns our test double instead of touching the network."""
    fake = types.ModuleType("huggingface_hub")
    fake.hf_hub_download = download_impl  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)


def test_try_fetch_hub_adapter_config_returns_none_on_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Network / auth failures must surface as ``None`` so the caller can
    fall back to the full-model branch instead of crashing."""

    def _boom(repo_id: str, filename: str) -> str:
        del repo_id, filename
        raise RuntimeError("simulated 401 / 404 from the hub")

    _install_fake_huggingface_hub(monkeypatch, _boom)
    assert _try_fetch_hub_adapter_config("user/some-repo") is None


def test_try_fetch_hub_adapter_config_returns_path_when_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Happy-path: the helper returns the local path the hub mirrored to."""
    fake_local = tmp_path / "adapter_config.json"
    fake_local.write_text(json.dumps({"base_model_name_or_path": "VietAI/vit5-base"}))

    def _ok(repo_id: str, filename: str) -> str:
        del repo_id
        assert filename == "adapter_config.json"
        return str(fake_local)

    _install_fake_huggingface_hub(monkeypatch, _ok)
    out = _try_fetch_hub_adapter_config("user/some-repo")
    assert out == fake_local
