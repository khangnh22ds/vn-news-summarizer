"""Process-wide singleton wrapping :class:`ViT5Summarizer`.

The model itself lazy-loads its weights on the first :meth:`summarize`
call; this helper just makes sure the same wrapper instance is reused
across requests so we don't reload the ~1 GB base model + ~7 MB adapter
per call.
"""

from __future__ import annotations

import threading
from typing import Final

from loguru import logger
from vn_news_common.settings import get_settings
from vn_news_inference import GenerationConfig, ViT5Summarizer

_LOCK: Final[threading.Lock] = threading.Lock()
_INSTANCE: ViT5Summarizer | None = None


def get_summarizer() -> ViT5Summarizer:
    """Return the cached :class:`ViT5Summarizer` for the current process."""
    global _INSTANCE  # noqa: PLW0603 — module-level cache by design
    if _INSTANCE is not None:
        return _INSTANCE
    with _LOCK:
        if _INSTANCE is None:
            settings = get_settings()
            logger.info("creating ViT5Summarizer for model_path={}", settings.model_path)
            _INSTANCE = ViT5Summarizer(
                settings.model_path,
                generation=GenerationConfig(),
            )
    return _INSTANCE


def reset_summarizer() -> None:
    """Drop the cached summarizer (test hook)."""
    global _INSTANCE  # noqa: PLW0603
    with _LOCK:
        _INSTANCE = None
