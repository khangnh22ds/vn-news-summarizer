"""Text normalization + similarity helpers."""

from __future__ import annotations

import re
import unicodedata

from simhash import Simhash

_WS_RE = re.compile(r"\s+")
_BOILERPLATE_RE = re.compile(
    r"(?im)^\s*(?:đọc thêm|xem thêm|tags?:|từ khóa:|liên quan:|chia sẻ).*$",
)


def normalize_text(text: str) -> str:
    """NFC-normalize, strip boilerplate lines, collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _BOILERPLATE_RE.sub("", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def word_count(text: str) -> int:
    """Whitespace-split word count (good enough for Vietnamese summary length)."""
    return len(text.split()) if text else 0


def simhash64(text: str) -> int:
    """Return a 64-bit SimHash of the input text as a *signed* int.

    ``simhash.Simhash`` returns an unsigned 64-bit value, but SQLite's
    INTEGER column is signed 64-bit (max 2**63 - 1), so we re-interpret
    the bits as signed two's-complement before returning. The
    :func:`hamming` helper masks back to 64 bits, so XOR distance is
    unaffected.
    """
    if not text:
        return 0
    val = int(Simhash(text).value) & ((1 << 64) - 1)
    if val >= (1 << 63):
        val -= 1 << 64
    return val


def hamming(a: int, b: int) -> int:
    """Hamming distance between two 64-bit SimHash values."""
    return bin((a ^ b) & ((1 << 64) - 1)).count("1")


def is_near_duplicate(a: int, b: int, *, threshold: int = 3) -> bool:
    """Return True if two simhash values are within ``threshold`` bits."""
    return hamming(a, b) <= threshold
