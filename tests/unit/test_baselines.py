"""Unit tests for the extractive baselines."""

from __future__ import annotations

import numpy as np
import pytest
from vn_news_training import (
    ExtractiveSummarizer,
    SummarizerConfig,
    sent_tokenize,
    word_tokenize,
)
from vn_news_training.baselines.extractive import (
    _build_lexrank_matrix,
    _build_textrank_matrix,
    _power_iteration,
    _word_overlap_similarity,
)

ARTICLE = (
    "Hôm 12/04, Bộ Y tế cho biết TP HCM ghi nhận 1.234 ca mắc Covid-19 mới. "
    "Chủ tịch Nguyễn Văn A chỉ đạo triển khai 5 biện pháp cấp bách để phòng "
    "dịch tại các quận trung tâm. Các bệnh viện tuyến cuối được yêu cầu sẵn "
    "sàng giường bệnh và oxy y tế. Sở Y tế thành phố sẽ giám sát việc tuân "
    "thủ tại các khu công nghiệp và trường học. Người dân được khuyến cáo "
    "đeo khẩu trang khi đến nơi đông người."
)


@pytest.mark.parametrize("baseline", ["lexrank", "textrank"])
def test_summarize_returns_subset_of_input(baseline: str) -> None:
    summarizer = ExtractiveSummarizer(
        SummarizerConfig(name=baseline, max_words=40, max_sentences=2)  # type: ignore[arg-type]
    )
    summary = summarizer.summarize(ARTICLE)
    assert summary, "summarizer must return non-empty output"
    # Sentences in the summary should appear (verbatim) in the source.
    for sent in sent_tokenize(summary):
        assert sent in ARTICLE, f"hallucinated sentence: {sent!r}"


def test_max_words_cap_is_enforced() -> None:
    summarizer = ExtractiveSummarizer(
        SummarizerConfig(name="lexrank", max_words=15, max_sentences=4)
    )
    summary = summarizer.summarize(ARTICLE)
    assert len(summary.split()) <= 15


def test_empty_input_returns_empty() -> None:
    summarizer = ExtractiveSummarizer()
    assert summarizer.summarize("") == ""
    assert summarizer.summarize("   \n") == ""


def test_single_sentence_input_returns_truncated() -> None:
    text = "Đây là một câu duy nhất với một số từ."
    summarizer = ExtractiveSummarizer(SummarizerConfig(max_words=4))
    out = summarizer.summarize(text)
    assert len(out.split()) <= 4
    assert out.split()[0] in text


def test_word_overlap_similarity_is_symmetric_and_zero_for_disjoint() -> None:
    a = ["hôm", "nay", "trời", "đẹp"]
    b = ["hôm", "qua", "mưa", "to"]
    assert _word_overlap_similarity(a, b) == _word_overlap_similarity(b, a)
    assert _word_overlap_similarity(a, []) == 0.0
    assert _word_overlap_similarity(["x"], ["y"]) == 0.0


def test_lexrank_matrix_is_zero_diagonal() -> None:
    sentences = [["a", "b", "c"], ["a", "b", "d"], ["x", "y"]]
    m = _build_lexrank_matrix(sentences, threshold=0.0)
    assert m.shape == (3, 3)
    assert np.allclose(np.diag(m), 0.0)


def test_textrank_matrix_is_symmetric() -> None:
    sentences = [["a", "b"], ["b", "c"], ["c", "a"]]
    m = _build_textrank_matrix(sentences)
    assert np.allclose(m, m.T)


def test_power_iteration_converges_to_uniform_on_uniform_input() -> None:
    n = 4
    matrix = np.ones((n, n)) - np.eye(n)
    scores = _power_iteration(matrix)
    assert pytest.approx(scores.sum(), abs=1e-2) == 1.0
    # All scores equal on a fully connected uniform graph.
    assert np.allclose(scores, scores[0], atol=1e-4)


def test_power_iteration_handles_empty_matrix() -> None:
    scores = _power_iteration(np.zeros((0, 0)))
    assert scores.shape == (0,)


def test_tokenizer_fallback_path() -> None:
    # Force the regex fallback (no underthesea) — must still work.
    sents = sent_tokenize("Câu một. Câu hai? Câu ba!", use_underthesea=False)
    assert sents == ["Câu một.", "Câu hai?", "Câu ba!"]
    words = word_tokenize("Tôi đi học", use_underthesea=False)
    assert words == ["Tôi", "đi", "học"]
