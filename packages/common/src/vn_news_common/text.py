"""Text normalization + similarity helpers."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from simhash import Simhash

# The upstream ``simhash`` library falls back to a slow path for any per-feature
# weight above :attr:`Simhash.large_weight_cutoff` (default 50). That slow path
# multiplies a ``uint8`` bitarray by the Python int weight, which raises
# ``OverflowError: Python integer N out of bounds for uint8`` on numpy >= 1.24
# (enforcing strict dtype casting). Long pages with a heavily repeated n-gram
# (e.g. a Thanh Nien table listing "Phường" hundreds of times) trigger this.
# We pre-tokenise exactly like the library and cap weights to stay on the fast
# ``batch`` path.
_SIMHASH_WEIGHT_CAP = 50
_SIMHASH_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fcc]+", re.UNICODE)
_SIMHASH_NGRAM_WIDTH = 4

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


def _simhash_features(text: str) -> dict[str, int]:
    """Tokenise ``text`` the same way ``Simhash.build_by_text`` does, but cap
    per-feature weights to dodge the upstream uint8 overflow.

    Mirrors ``Simhash._tokenize`` + ``_slide``: lowercase, concatenate regex
    matches, emit a ``Counter`` of character 4-grams.
    """
    lowered = text.lower()
    joined = "".join(_SIMHASH_TOKEN_RE.findall(lowered))
    if not joined:
        return {}
    if len(joined) < _SIMHASH_NGRAM_WIDTH:
        return {joined: 1}
    counts = Counter(
        joined[i : i + _SIMHASH_NGRAM_WIDTH] for i in range(len(joined) - _SIMHASH_NGRAM_WIDTH + 1)
    )
    return {feature: min(weight, _SIMHASH_WEIGHT_CAP) for feature, weight in counts.items()}


def simhash64(text: str) -> int:
    """Return a 64-bit SimHash of the input text as a *signed* int.

    ``simhash.Simhash`` returns an unsigned 64-bit value, but SQLite's
    INTEGER column is signed 64-bit (max 2**63 - 1), so we re-interpret
    the bits as signed two's-complement before returning. The
    :func:`hamming` helper masks back to 64 bits, so XOR distance is
    unaffected.

    Per-feature weights are capped at :data:`_SIMHASH_WEIGHT_CAP` to avoid
    ``OverflowError`` from the upstream library on heavily repetitive pages.
    """
    if not text:
        return 0
    features = _simhash_features(text)
    if not features:
        return 0
    val = int(Simhash(features, f=64).value) & ((1 << 64) - 1)
    if val >= (1 << 63):
        val -= 1 << 64
    return val


def hamming(a: int, b: int) -> int:
    """Hamming distance between two 64-bit SimHash values."""
    return bin((a ^ b) & ((1 << 64) - 1)).count("1")


def is_near_duplicate(a: int, b: int, *, threshold: int = 3) -> bool:
    """Return True if two simhash values are within ``threshold`` bits."""
    return hamming(a, b) <= threshold
