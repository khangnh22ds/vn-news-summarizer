"""Unit tests for the YAML training config loader."""

from __future__ import annotations

import importlib
import os
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
from vn_news_training.config import MlflowConfig
from vn_news_training.finetune import _configure_mlflow

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_load_default_vit5_config_round_trips() -> None:
    cfg = load_finetune_config(REPO_ROOT / "configs/training/vit5_base_v1.yaml")
    assert isinstance(cfg, FinetuneConfig)
    assert cfg.run_name == "vit5-base-news-v1"
    assert cfg.model.base == "VietAI/vit5-base"
    assert cfg.model.max_input_length == 1024
    assert cfg.dataset.text_column == "content_text"
    assert cfg.dataset.target_column == "summary"
    assert cfg.peft.enabled is True
    assert cfg.peft.r == 16
    assert cfg.training.num_train_epochs >= 1
    assert cfg.training.eval_strategy == "epoch"
    assert cfg.training.fp16 is True
    assert "mlflow" in cfg.training.report_to
    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.peft, PeftConfig)
    assert isinstance(cfg.training, TrainingConfig)
    assert isinstance(cfg.dataset, DatasetConfig)


def test_load_finetune_config_uses_defaults_for_missing_sections(tmp_path: Path) -> None:
    minimal = tmp_path / "minimal.yaml"
    minimal.write_text("run_name: my-run\n", encoding="utf-8")
    cfg = load_finetune_config(minimal)
    assert cfg.run_name == "my-run"
    # Unspecified sections fall back to their dataclass defaults.
    assert cfg.model.base == "VietAI/vit5-base"
    assert cfg.training.num_train_epochs > 0
    assert cfg.peft.enabled is True


def test_load_finetune_config_ignores_unknown_keys(tmp_path: Path) -> None:
    """Unknown keys in YAML shouldn't explode the loader — just be dropped."""
    yaml_path = tmp_path / "extra.yaml"
    yaml_path.write_text(
        "run_name: my-run\n"
        "mystery_field: 42\n"
        "training:\n  num_train_epochs: 2\n  mystery_subkey: true\n",
        encoding="utf-8",
    )
    cfg = load_finetune_config(yaml_path)
    assert cfg.run_name == "my-run"
    assert cfg.training.num_train_epochs == 2


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


class _FakeMlflow:
    def __init__(self) -> None:
        self.calls: dict[str, str] = {}

    def set_tracking_uri(self, uri: str) -> None:
        self.calls["tracking_uri"] = uri

    def set_experiment(self, name: str) -> None:
        self.calls["experiment"] = name


def test_configure_mlflow_sets_env_and_mlflow_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """``run_finetune`` must wire YAML mlflow config to env vars before the
    HuggingFace ``MLflowCallback`` reads them; otherwise runs leak to the
    default experiment with the env-default tracking URI."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    fake = _FakeMlflow()
    monkeypatch.setattr(importlib, "import_module", lambda name: fake)

    _configure_mlflow(MlflowConfig(experiment_name="my-exp", tracking_uri="file:./custom-mlruns"))

    assert os.environ["MLFLOW_TRACKING_URI"] == "file:./custom-mlruns"
    assert os.environ["MLFLOW_EXPERIMENT_NAME"] == "my-exp"
    assert fake.calls == {
        "tracking_uri": "file:./custom-mlruns",
        "experiment": "my-exp",
    }


def test_configure_mlflow_respects_existing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-existing MLFLOW_* env vars (e.g. set by the user in Colab) take
    precedence over the YAML config — we only fall back to the config."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "user-override")
    fake = _FakeMlflow()
    monkeypatch.setattr(importlib, "import_module", lambda name: fake)

    _configure_mlflow(MlflowConfig(experiment_name="from-yaml", tracking_uri="file:./yaml-mlruns"))

    assert fake.calls == {
        "tracking_uri": "https://mlflow.example.com",
        "experiment": "user-override",
    }
