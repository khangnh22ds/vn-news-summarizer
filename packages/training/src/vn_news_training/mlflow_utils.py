"""Thin MLflow tracking helpers (local file backend).

We default to ``mlruns/`` at the repo root, matching the gitignored
output convention. Callers that want a remote tracking server can set
``MLFLOW_TRACKING_URI`` and the helpers will respect it.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow


def _resolve_tracking_uri(default_dir: Path | str = "mlruns") -> str:
    """Honour ``MLFLOW_TRACKING_URI`` if set, else point at ``./mlruns``."""
    explicit = os.environ.get("MLFLOW_TRACKING_URI")
    if explicit:
        return explicit
    return Path(default_dir).absolute().as_uri().replace("file://", "file:")


@contextmanager
def mlflow_run(
    *,
    experiment: str = "baselines",
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> Iterator[Any]:
    """Open an MLflow run with the local file backend.

    Yields the active ``mlflow.ActiveRun`` so callers can ``log_param`` /
    ``log_metric`` / ``log_artifact`` directly.
    """
    mlflow.set_tracking_uri(_resolve_tracking_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
        yield run


def log_metrics(metrics: dict[str, float | int]) -> None:
    """Log a flat dict of numeric metrics to the active run."""
    for k, v in metrics.items():
        mlflow.log_metric(k, float(v))


def log_params(params: dict[str, object]) -> None:
    """Log a flat dict of parameters (cast to ``str``)."""
    for k, v in params.items():
        mlflow.log_param(k, str(v))
