"""Baselines, training pipeline, and evaluation harness.

Phase 3 (TICKET-004) lands the extractive baselines + ROUGE/BERTScore
eval harness. Phase 4 (TICKET-005) will add fine-tuning on top.
"""

from __future__ import annotations

from .baselines import (
    BaselineName,
    ExtractiveSummarizer,
    SummarizerConfig,
    sent_tokenize,
    word_tokenize,
)
from .dataset import Example, SplitName, load_dataset, load_split
from .eval import (
    EvalResult,
    RougeAggregate,
    compute_bertscore,
    compute_rouge,
    evaluate_predictions,
)
from .mlflow_utils import log_metrics, log_params, mlflow_run

__version__ = "0.1.0"
__all__ = [
    "BaselineName",
    "EvalResult",
    "Example",
    "ExtractiveSummarizer",
    "RougeAggregate",
    "SplitName",
    "SummarizerConfig",
    "__version__",
    "compute_bertscore",
    "compute_rouge",
    "evaluate_predictions",
    "load_dataset",
    "load_split",
    "log_metrics",
    "log_params",
    "mlflow_run",
    "sent_tokenize",
    "word_tokenize",
]
