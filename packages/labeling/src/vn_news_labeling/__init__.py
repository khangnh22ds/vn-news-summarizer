"""LLM-assisted summarization labeling pipeline.

See :mod:`vn_news_labeling.prompt` (template loader),
:mod:`vn_news_labeling.vertex_client` (Vertex AI Gemini wrapper),
:mod:`vn_news_labeling.qc` (deterministic QC battery),
:mod:`vn_news_labeling.pipeline` (orchestrator), and
:mod:`vn_news_labeling.dataset` (JSONL export + dataset versioning).
"""

from __future__ import annotations

from .dataset import DatasetStats, build_dataset_version
from .pipeline import LabelStats, label_batch
from .prompt import LabelOutput, Prompt, parse_label_json
from .qc import QcResult, run_qc
from .vertex_client import (
    GenerationParams,
    VertexLabeler,
    VertexLLMError,
    VertexTransientError,
)

__version__ = "0.1.0"
__all__ = [
    "DatasetStats",
    "GenerationParams",
    "LabelOutput",
    "LabelStats",
    "Prompt",
    "QcResult",
    "VertexLLMError",
    "VertexLabeler",
    "VertexTransientError",
    "__version__",
    "build_dataset_version",
    "label_batch",
    "parse_label_json",
    "run_qc",
]
