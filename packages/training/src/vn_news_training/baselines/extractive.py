"""LexRank and TextRank extractive summarizers tuned for Vietnamese.

Both use the same recipe — sentence segmentation → bag-of-words /
TF-IDF vectorization → similarity graph → power-iteration PageRank — and
differ only in the similarity function:

- **TextRank**: symmetric word-overlap similarity (Mihalcea & Tarau 2004).
- **LexRank**: cosine similarity of TF-IDF vectors (Erkan & Radev 2004).

We keep the implementations small and dependency-light (numpy +
scikit-learn) so they're easy to read, test, and swap out. Vietnamese
tokenization comes from :mod:`underthesea` via
:mod:`vn_news_training.baselines.tokenizer`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .tokenizer import sent_tokenize, word_tokenize

BaselineName = Literal["lexrank", "textrank"]


@dataclass(slots=True)
class SummarizerConfig:
    """Knobs shared by both baselines."""

    name: BaselineName = "lexrank"
    max_words: int = 70
    """Hard cap on the output summary length (in words)."""
    max_sentences: int = 4
    """Soft cap on the number of selected sentences."""
    damping: float = 0.85
    """PageRank damping factor."""
    threshold: float = 0.0
    """Edges below this similarity weight are dropped (only used by LexRank)."""
    use_underthesea: bool | None = None
    """Override sentence/word tokenization. ``None`` = auto-detect."""


def _word_overlap_similarity(a: list[str], b: list[str]) -> float:
    """Mihalcea-Tarau sentence similarity: overlapping words / log-length."""
    if not a or not b:
        return 0.0
    overlap = len(set(a) & set(b))
    if overlap == 0:
        return 0.0
    denom = float(np.log(1 + len(a)) + np.log(1 + len(b)))
    if denom <= 0.0:
        return 0.0
    return overlap / denom


def _build_lexrank_matrix(sentence_tokens: Sequence[Sequence[str]], threshold: float) -> np.ndarray:
    """Cosine similarity over TF-IDF vectors, with optional thresholding."""
    docs = [" ".join(toks) for toks in sentence_tokens]
    if not docs:
        return np.zeros((0, 0), dtype=np.float64)
    vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=True)
    try:
        tfidf = vec.fit_transform(docs)
    except ValueError:
        # All-stopword / empty corpus; fall back to identity weights.
        return np.eye(len(docs), dtype=np.float64)
    sim: np.ndarray = cosine_similarity(tfidf).astype(np.float64)
    if threshold > 0.0:
        sim = np.where(sim >= threshold, sim, 0.0)
    np.fill_diagonal(sim, 0.0)
    return sim


def _build_textrank_matrix(sentence_tokens: Sequence[Sequence[str]]) -> np.ndarray:
    """Symmetric word-overlap similarity (TextRank)."""
    n = len(sentence_tokens)
    sim = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            w = _word_overlap_similarity(list(sentence_tokens[i]), list(sentence_tokens[j]))
            sim[i, j] = w
            sim[j, i] = w
    return sim


def _power_iteration(
    matrix: np.ndarray,
    *,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Standard PageRank power iteration on a similarity matrix."""
    n = matrix.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    row_sum = matrix.sum(axis=1, keepdims=True)
    # Replace zero rows with a uniform distribution to avoid NaN.
    row_sum[row_sum == 0.0] = 1.0
    transition = matrix / row_sum
    scores = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = (1.0 - damping) / n
    for _ in range(max_iter):
        new_scores = teleport + damping * (transition.T @ scores)
        if np.linalg.norm(new_scores - scores, ord=1) < tol:
            scores = new_scores
            break
        scores = new_scores
    return scores


class ExtractiveSummarizer:
    """LexRank / TextRank summarizer.

    Stateless aside from the configuration; safe to reuse across articles.
    """

    def __init__(self, cfg: SummarizerConfig | None = None) -> None:
        self.cfg = cfg or SummarizerConfig()

    def summarize(self, text: str) -> str:
        """Return an extractive summary of ``text``.

        The output is a substring concatenation of the highest-scoring
        sentences (in original order), capped at ``cfg.max_words`` words
        and ``cfg.max_sentences`` sentences.
        """
        sentences = sent_tokenize(text, use_underthesea=self.cfg.use_underthesea)
        if not sentences:
            return ""
        if len(sentences) == 1:
            return self._truncate(sentences[0])

        sentence_tokens = [
            [tok.lower() for tok in word_tokenize(s, use_underthesea=self.cfg.use_underthesea)]
            for s in sentences
        ]

        if self.cfg.name == "lexrank":
            matrix = _build_lexrank_matrix(sentence_tokens, threshold=self.cfg.threshold)
        else:
            matrix = _build_textrank_matrix(sentence_tokens)

        scores = _power_iteration(matrix, damping=self.cfg.damping)
        # Pick the top-K by score, then re-sort by original position so
        # the summary reads naturally.
        order_by_score = np.argsort(-scores)
        picked: list[int] = []
        for idx in order_by_score:
            picked.append(int(idx))
            if len(picked) >= self.cfg.max_sentences:
                break
        picked.sort()
        merged = " ".join(sentences[i] for i in picked)
        return self._truncate(merged)

    def _truncate(self, text: str) -> str:
        words = text.split()
        if len(words) <= self.cfg.max_words:
            return text.strip()
        return " ".join(words[: self.cfg.max_words]).strip()
