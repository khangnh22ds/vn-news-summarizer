"""Tokenize the labeled JSONL splits into a HuggingFace ``DatasetDict``.

The labeling pipeline writes raw JSONL records (see
``packages/labeling.dataset.build_dataset_version``); the trainer needs
``input_ids`` / ``attention_mask`` / ``labels`` tensors.

This module does the conversion using the model's tokenizer and the
column names from the training config, with truncation to the configured
input/target lengths.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Protocol

from .config import DatasetConfig, ModelConfig
from .dataset import load_split


class _Tokenizer(Protocol):
    """Subset of ``PreTrainedTokenizerBase`` we actually use."""

    def __call__(
        self,
        text: str | list[str],
        *,
        max_length: int = ...,
        truncation: bool = ...,
        padding: str | bool = ...,
    ) -> dict[str, Any]: ...


def _tokenize_record(
    rec_text: str,
    rec_summary: str,
    *,
    tokenizer: _Tokenizer,
    max_input_length: int,
    max_target_length: int,
) -> dict[str, list[int]]:
    enc = tokenizer(
        rec_text,
        max_length=max_input_length,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        rec_summary,
        max_length=max_target_length,
        truncation=True,
        padding=False,
    )
    return {
        "input_ids": list(enc["input_ids"]),
        "attention_mask": list(enc["attention_mask"]),
        "labels": list(labels["input_ids"]),
    }


def build_examples(
    *,
    dataset_cfg: DatasetConfig,
    model_cfg: ModelConfig,
    tokenizer: _Tokenizer,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, list[dict[str, list[int]]]]:
    """Tokenize all requested splits and return a plain Python dict.

    The output is intentionally not a ``datasets.DatasetDict`` so the
    function can run (and be unit-tested) without importing ``datasets``
    or ``torch``. The trainer wrapper converts it as needed.
    """
    dataset_dir = Path(dataset_cfg.root) / dataset_cfg.name
    out: dict[str, list[dict[str, list[int]]]] = {}
    for split in splits:
        examples = load_split(dataset_dir, split)  # type: ignore[arg-type]
        rows: list[dict[str, list[int]]] = []
        for ex in examples:
            text = getattr(ex, dataset_cfg.text_column, ex.content_text)
            target = getattr(ex, dataset_cfg.target_column, ex.summary)
            if not text or not target:
                continue
            rows.append(
                _tokenize_record(
                    str(text),
                    str(target),
                    tokenizer=tokenizer,
                    max_input_length=model_cfg.max_input_length,
                    max_target_length=model_cfg.max_target_length,
                )
            )
        out[split] = rows
    return out


def build_hf_dataset_dict(
    *,
    dataset_cfg: DatasetConfig,
    model_cfg: ModelConfig,
    tokenizer: _Tokenizer,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> Any:
    """Like :func:`build_examples` but returns a ``datasets.DatasetDict``.

    Lazy-imports ``datasets`` so unit tests don't pay the cost.
    """
    datasets_mod = importlib.import_module("datasets")
    raw = build_examples(
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        tokenizer=tokenizer,
        splits=splits,
    )
    return datasets_mod.DatasetDict(
        {name: datasets_mod.Dataset.from_list(rows) for name, rows in raw.items()}
    )
