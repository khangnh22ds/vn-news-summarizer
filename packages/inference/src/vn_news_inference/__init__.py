"""Model inference + Vertex AI LLM fallback.

TICKET-005 ships the ViT5 / LoRA loader. TICKET-006 will add the FastAPI
serving + Vertex fallback path on top.
"""

from __future__ import annotations

from .finetune_loader import GenerationConfig, ViT5Summarizer

__version__ = "0.1.0"
__all__ = [
    "GenerationConfig",
    "ViT5Summarizer",
    "__version__",
]
