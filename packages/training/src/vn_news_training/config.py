"""Strongly-typed loader for the training YAML config.

Mirrors ``configs/training/vit5_base_v1.yaml``. Kept light (no pydantic
runtime parsing) so it can be imported in CI without pulling extra deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DatasetConfig:
    name: str = "v1"
    root: str = "data/datasets"
    text_column: str = "content_text"
    target_column: str = "summary"


@dataclass(slots=True)
class ModelConfig:
    base: str = "VietAI/vit5-base"
    max_input_length: int = 1024
    max_target_length: int = 128


@dataclass(slots=True)
class TrainingConfig:
    output_dir: str = "models/vit5-news-v1"
    num_train_epochs: float = 4.0
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5.0e-5
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    fp16: bool = True
    bf16: bool = False
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "rougeL"
    greater_is_better: bool = True
    predict_with_generate: bool = True
    generation_num_beams: int = 4
    generation_max_length: int = 128
    logging_steps: int = 50
    save_total_limit: int = 2
    seed: int = 42
    report_to: list[str] = field(default_factory=lambda: ["mlflow"])


@dataclass(slots=True)
class PeftConfig:
    enabled: bool = True
    type: str = "lora"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q", "v"])


@dataclass(slots=True)
class EvalConfig:
    metrics: list[str] = field(default_factory=lambda: ["rouge1", "rouge2", "rougeL"])
    bertscore_model: str = "xlm-roberta-base"


@dataclass(slots=True)
class MlflowConfig:
    experiment_name: str = "vn-news-summarization"
    tracking_uri: str = "file:./mlruns"


@dataclass(slots=True)
class FinetuneConfig:
    """Top-level training config (matches the YAML schema 1:1)."""

    version: str = "1.0.0"
    run_name: str = "vit5-base-news-v1"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    peft: PeftConfig = field(default_factory=PeftConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)


def _build_dataclass(cls: type, raw: dict[str, Any] | None) -> Any:
    if raw is None:
        return cls()
    valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in raw.items() if k in valid})


def load_finetune_config(path: Path | str) -> FinetuneConfig:
    """Read a YAML config file into a typed :class:`FinetuneConfig`."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return FinetuneConfig(
        version=str(raw.get("version", "1.0.0")),
        run_name=str(raw.get("run_name", "vit5-base-news-v1")),
        dataset=_build_dataclass(DatasetConfig, raw.get("dataset")),
        model=_build_dataclass(ModelConfig, raw.get("model")),
        training=_build_dataclass(TrainingConfig, raw.get("training")),
        peft=_build_dataclass(PeftConfig, raw.get("peft")),
        evaluation=_build_dataclass(EvalConfig, raw.get("evaluation")),
        mlflow=_build_dataclass(MlflowConfig, raw.get("mlflow")),
    )
