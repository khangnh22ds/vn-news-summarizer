"""Unit tests for the JSONL → tokenized-dict preprocessing layer.

Uses a tiny fake tokenizer so we don't have to download or load any
real HuggingFace tokenizer (CI-friendly, fast).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from vn_news_training import DatasetConfig, ModelConfig
from vn_news_training.preprocess import build_examples


class _FakeTokenizer:
    """Whitespace tokenizer that fits the :class:`_Tokenizer` Protocol."""

    def __call__(
        self,
        text: str | list[str],
        *,
        max_length: int = 1024,
        truncation: bool = False,
        padding: str | bool = False,
    ) -> dict[str, Any]:
        del padding  # unused
        if isinstance(text, list):  # pragma: no cover — not exercised
            text = text[0]
        toks = text.split()
        if truncation and len(toks) > max_length:
            toks = toks[:max_length]
        ids = [hash(t) % 32_000 for t in toks]
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
        }


def _write_jsonl(path: Path, n: int, *, content_words: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = " ".join([f"w{i}" for i in range(content_words)])
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(
                json.dumps(
                    {
                        "article_id": i,
                        "source": "vnexpress",
                        "url": f"https://x/{i}",
                        "title": "T",
                        "category": "thoi-su",
                        "published_at": None,
                        "content_text": body,
                        "summary": "tiny summary",
                        "prompt_version": "1.0.0",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _make_dataset(tmp_path: Path, *, name: str = "v1", n_per_split: int = 2) -> Path:
    root = tmp_path / "data" / "datasets"
    out = root / name
    for split in ("train", "val", "test"):
        _write_jsonl(out / f"{split}.jsonl", n=n_per_split)
    return root


def test_build_examples_returns_tokenized_records(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path, n_per_split=3)
    out = build_examples(
        dataset_cfg=DatasetConfig(name="v1", root=str(root)),
        model_cfg=ModelConfig(max_input_length=64, max_target_length=8),
        tokenizer=_FakeTokenizer(),
    )
    assert set(out.keys()) == {"train", "val", "test"}
    assert all(len(rows) == 3 for rows in out.values())
    sample = out["train"][0]
    assert "input_ids" in sample and "attention_mask" in sample and "labels" in sample
    assert isinstance(sample["input_ids"], list)
    assert len(sample["input_ids"]) == len(sample["attention_mask"])


def test_build_examples_truncates_to_max_input_length(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path, n_per_split=1)
    # Source has 6 words; cap input at 3.
    out = build_examples(
        dataset_cfg=DatasetConfig(name="v1", root=str(root)),
        model_cfg=ModelConfig(max_input_length=3, max_target_length=8),
        tokenizer=_FakeTokenizer(),
    )
    sample = out["train"][0]
    assert len(sample["input_ids"]) == 3


def test_build_examples_skips_empty_text_or_summary(tmp_path: Path) -> None:
    root = tmp_path / "data" / "datasets"
    out_dir = root / "v1"
    out_dir.mkdir(parents=True)
    rows = [
        # valid
        {
            "article_id": 1,
            "source": "x",
            "url": "y",
            "title": "T",
            "category": None,
            "published_at": None,
            "content_text": "hello world",
            "summary": "ok",
            "prompt_version": "1.0.0",
        },
        # empty summary -> skipped
        {
            "article_id": 2,
            "source": "x",
            "url": "y",
            "title": "T",
            "category": None,
            "published_at": None,
            "content_text": "hello world",
            "summary": "",
            "prompt_version": "1.0.0",
        },
    ]
    for split in ("train", "val", "test"):
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    out = build_examples(
        dataset_cfg=DatasetConfig(name="v1", root=str(root)),
        model_cfg=ModelConfig(max_input_length=64, max_target_length=8),
        tokenizer=_FakeTokenizer(),
    )
    # Only the valid row survives in each split.
    assert all(len(v) == 1 for v in out.values())


def test_build_examples_subset_of_splits(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path, n_per_split=1)
    out = build_examples(
        dataset_cfg=DatasetConfig(name="v1", root=str(root)),
        model_cfg=ModelConfig(max_input_length=64, max_target_length=8),
        tokenizer=_FakeTokenizer(),
        splits=("train",),
    )
    assert list(out.keys()) == ["train"]


def test_build_examples_missing_split_raises(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    (root / "v1").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        build_examples(
            dataset_cfg=DatasetConfig(name="v1", root=str(root)),
            model_cfg=ModelConfig(max_input_length=64, max_target_length=8),
            tokenizer=_FakeTokenizer(),
            splits=("train",),
        )
