"""Vietnamese tokenizers used by the extractive baselines.

Wraps :mod:`underthesea` with a regex fallback. Underthesea uses C
extensions that occasionally segfault on interpreter shutdown, so the
import is lazy + cacheable and can be disabled via the environment
variable ``VN_NEWS_USE_UNDERTHESEA=0`` (used by the test suite).
"""

from __future__ import annotations

import importlib
import os
import re
from collections.abc import Callable
from typing import Any

_SENT_FALLBACK = re.compile(r"(?<=[\.!?…])\s+")
_WORD_FALLBACK = re.compile(r"[\wÀ-ỹ]+", flags=re.UNICODE)

# Lazy-cached references to the underthesea callables. ``None`` means
# "never tried", ``False`` means "tried and not available".
_ut_cache: dict[str, Callable[..., Any] | bool] = {}


def _ut_enabled() -> bool:
    """``True`` unless the user has set ``VN_NEWS_USE_UNDERTHESEA=0``."""
    return os.environ.get("VN_NEWS_USE_UNDERTHESEA", "1") != "0"


def _load_underthesea() -> tuple[Callable[..., Any] | None, Callable[..., Any] | None]:
    """Import + cache underthesea on first use. Returns ``(sent, word)``
    tokenizers, or ``(None, None)`` if the import failed or is disabled.
    """
    if "sent" in _ut_cache:
        sent = _ut_cache["sent"]
        word = _ut_cache["word"]
        return (
            sent if callable(sent) else None,
            word if callable(word) else None,
        )
    if not _ut_enabled():
        _ut_cache["sent"] = False
        _ut_cache["word"] = False
        return None, None
    try:
        # Loaded via importlib so the heavy CRF C-extension stays out of
        # the import graph until actually needed.
        ut = importlib.import_module("underthesea")
    except Exception:
        _ut_cache["sent"] = False
        _ut_cache["word"] = False
        return None, None
    ut_sent = ut.sent_tokenize
    ut_word = ut.word_tokenize
    _ut_cache["sent"] = ut_sent
    _ut_cache["word"] = ut_word
    return ut_sent, ut_word


def sent_tokenize(text: str, *, use_underthesea: bool | None = None) -> list[str]:
    """Split ``text`` into sentences. Empty input returns ``[]``."""
    if not text or not text.strip():
        return []
    use_ut = True if use_underthesea is None else use_underthesea
    if use_ut:
        ut_sent, _ = _load_underthesea()
        if ut_sent is not None:
            return [s.strip() for s in ut_sent(text) if s and s.strip()]
    return [s.strip() for s in _SENT_FALLBACK.split(text.strip()) if s.strip()]


def word_tokenize(text: str, *, use_underthesea: bool | None = None) -> list[str]:
    r"""Split ``text`` into Vietnamese word tokens.

    Underthesea returns multi-word units (e.g. ``"Hà Nội"``) when
    available, otherwise we fall back to unicode-aware ``\w+`` extraction.
    """
    if not text or not text.strip():
        return []
    use_ut = True if use_underthesea is None else use_underthesea
    if use_ut:
        _, ut_word = _load_underthesea()
        if ut_word is not None:
            return [t for t in ut_word(text) if t.strip()]
    return _WORD_FALLBACK.findall(text)
