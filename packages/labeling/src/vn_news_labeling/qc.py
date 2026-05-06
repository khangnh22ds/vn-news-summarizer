"""Quality-control checks for LLM-generated summaries.

The goal is to catch the most common failure modes of LLM labeling
*without* a heavy NLI model:

1. **Length** — must be inside ``[min_words, max_words]``.
2. **Sentence count** — must be inside ``[min_sentences, max_sentences]``.
3. **Number/date faithfulness** — every digit-bearing token in the
   summary must appear in the source via one of three strategies:
   literal substring, punctuation-collapsed substring (so ``2-0`` matches
   ``2 0`` and vice-versa), or, for tokens that contain digit-groups of
   length ≥ 3 (years, large numbers), every such group must appear
   individually in the source.
4. **Entity faithfulness** — every Title-Cased multi-token name in the
   summary must appear in the source via literal substring, punctuation-
   collapsed substring, fuzzy ratio ≥ ``entity_fuzzy_min_ratio``, or any
   single token of length ≥ 4 appearing literally (catches ``HLV Simeone``
   when the source uses bare ``Simeone``).
5. **Refusal** — if the LLM populated ``refusal_reason``, fail QC.

The source corpus passed in by the pipeline is ``title + content_text``
rather than ``content_text`` alone; titles are part of the trusted
metadata pulled from the same article and frequently carry the league /
subject names referenced in summaries (e.g. "Premier League").

All checks are deterministic and CPU-only. NLI factuality is deferred
to a later ticket.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from .prompt import LabelOutput, QcCfg

# Token used to split summary into sentences. Keeps it simple — Vietnamese
# news summaries rarely contain abbreviations that fool this regex.
_SENT_SPLIT = re.compile(r"(?<=[\.!?…])\s+")
# A "number-like" token: contains at least one digit. Catches "20.000",
# "12/04", "3,5%", "USD 100", etc.
_NUMERIC = re.compile(r"\S*\d[\S]*")
# Punctuation/whitespace characters that we collapse to a single space
# when comparing tokens against the source. Covers regular hyphens,
# en-/em-dashes, common bracketing punctuation, and slashes.
_PUNCT_RE = re.compile(r"[\s\-\u2013\u2014\.\,;:\(\)\[\]\{\}/]+")
# Digit groups long enough to be meaningful on their own (years like
# ``2024``, large numbers like ``1234``). Two-digit groups are too noisy
# to fall back on — a hallucinated "42" would otherwise pass against any
# article containing the digits ``4`` and ``2`` somewhere.
_DIGIT_GROUP = re.compile(r"\d{3,}")
# A "named entity" candidate: one or more Title-Cased Vietnamese words.
# Allows accented capitals (e.g. "Đà Nẵng", "Nguyễn Phú Trọng").
_TITLE_CHARS = "A-ZĐÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ"
_ENTITY = re.compile(rf"(?:[{_TITLE_CHARS}][\wÀ-ỹ]*(?:\s+[{_TITLE_CHARS}][\wÀ-ỹ]*)+)")


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()


def _split_sentences(s: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(_norm(s)) if p.strip()]
    return parts


def _word_count(s: str) -> int:
    return len(_norm(s).split())


def _extract_numerics(s: str) -> list[str]:
    out: list[str] = []
    for m in _NUMERIC.finditer(s):
        token = m.group(0).strip(".,;:%()[]{}")
        if token:
            out.append(token)
    return out


def _collapse_punct(s: str) -> str:
    """Collapse separator punctuation/whitespace to a single space.

    Used so that ``2-0`` and ``2 0`` (or ``2-0`` with an en dash) are
    treated as the same surface form when searching for a number inside
    the source text.
    """
    return _PUNCT_RE.sub(" ", s).strip()


def _extract_entities(s: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in _ENTITY.finditer(s):
        name = m.group(0).strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


@dataclass(slots=True)
class QcResult:
    """Result of running QC on a single summary."""

    passed: bool
    reasons: list[str] = field(default_factory=list)
    word_count: int = 0
    sentence_count: int = 0
    missing_numbers: list[str] = field(default_factory=list)
    missing_entities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "missing_numbers": list(self.missing_numbers),
            "missing_entities": list(self.missing_entities),
        }


def _contains_numeric(source: str, token: str) -> bool:
    """Best-effort "is this number supported by the source?" check.

    Strategies, tried in order:
    1. Literal substring match.
    2. Punctuation-collapsed substring match (handles ``2-0`` vs ``2 0``
       and en-/em-dash variants in date ranges).
    3. Every digit-group of length ≥ 3 in the token must appear
       individually in the source (handles ``2019-2024`` against an
       article that lists the years separately).
    """
    if token in source:
        return True
    collapsed_source = _collapse_punct(source)
    collapsed_token = _collapse_punct(token)
    if collapsed_token and collapsed_token in collapsed_source:
        return True
    groups = _DIGIT_GROUP.findall(token)
    return bool(groups and all(g in source for g in groups))


def _contains_entity(source_norm: str, entity: str, *, min_ratio: float) -> bool:
    if entity in source_norm:
        return True
    collapsed_source = _collapse_punct(source_norm)
    collapsed_entity = _collapse_punct(entity)
    if collapsed_entity and collapsed_entity in collapsed_source:
        return True
    score = fuzz.partial_ratio(entity, source_norm)
    if score >= min_ratio * 100:
        return True
    # Last-name fallback: ``HLV Simeone`` should match a source that
    # mentions the bare surname ``Simeone``. Only tokens long enough to be
    # meaningful (>= 4 characters) qualify, so common helpers like ``HLV``
    # or ``CLB`` cannot single-handedly carry the match.
    return any(len(tok) >= 4 and tok in source_norm for tok in entity.split())


def run_qc(
    *,
    output: LabelOutput,
    source_text: str,
    cfg: QcCfg,
) -> QcResult:
    """Apply the QC battery and return a :class:`QcResult`."""
    reasons: list[str] = []
    summary = _norm(output.summary)
    source_norm = _norm(source_text)

    if output.refusal_reason:
        reasons.append(f"llm_refusal:{output.refusal_reason}")

    wc = _word_count(summary)
    if wc < cfg.min_words:
        reasons.append(f"too_short:{wc}<{cfg.min_words}")
    if wc > cfg.max_words:
        reasons.append(f"too_long:{wc}>{cfg.max_words}")

    sentences = _split_sentences(summary)
    sc = len(sentences)
    if sc < cfg.min_sentences:
        reasons.append(f"too_few_sentences:{sc}<{cfg.min_sentences}")
    if sc > cfg.max_sentences:
        reasons.append(f"too_many_sentences:{sc}>{cfg.max_sentences}")

    missing_numbers = [
        n for n in _extract_numerics(summary) if not _contains_numeric(source_norm, n)
    ]
    if missing_numbers:
        reasons.append(f"unsupported_numbers:{','.join(missing_numbers[:5])}")

    missing_entities = [
        e
        for e in _extract_entities(summary)
        if not _contains_entity(source_norm, e, min_ratio=cfg.entity_fuzzy_min_ratio)
    ]
    if missing_entities:
        reasons.append(f"unsupported_entities:{','.join(missing_entities[:5])}")

    passed = not reasons
    return QcResult(
        passed=passed,
        reasons=reasons,
        word_count=wc,
        sentence_count=sc,
        missing_numbers=missing_numbers,
        missing_entities=missing_entities,
    )
