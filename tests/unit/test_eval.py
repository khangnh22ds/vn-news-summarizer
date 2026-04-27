"""Unit tests for the eval harness (ROUGE; BERTScore is integration-only)."""

from __future__ import annotations

import pytest
from vn_news_training import (
    EvalResult,
    compute_rouge,
    evaluate_predictions,
)


def test_compute_rouge_perfect_match_is_one() -> None:
    preds = ["hôm nay trời đẹp"]
    refs = ["hôm nay trời đẹp"]
    agg = compute_rouge(preds, refs)
    assert agg.n == 1
    assert agg.rouge1 == pytest.approx(1.0)
    assert agg.rouge2 == pytest.approx(1.0)
    assert agg.rougeL == pytest.approx(1.0)


def test_compute_rouge_disjoint_is_zero() -> None:
    preds = ["abc def"]
    refs = ["xyz uvw"]
    agg = compute_rouge(preds, refs)
    assert agg.rouge1 == 0.0
    assert agg.rougeL == 0.0


def test_compute_rouge_partial_overlap_is_between_zero_and_one() -> None:
    preds = ["hôm nay trời đẹp"]
    refs = ["hôm qua trời đẹp"]
    agg = compute_rouge(preds, refs)
    assert 0.0 < agg.rouge1 < 1.0


def test_compute_rouge_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        compute_rouge(["a"], ["a", "b"])


def test_compute_rouge_empty_inputs_returns_zero() -> None:
    agg = compute_rouge([], [])
    assert agg.n == 0
    assert agg.rouge1 == 0.0


def test_evaluate_predictions_no_bertscore() -> None:
    result = evaluate_predictions(["a b c"], ["a b c"], use_bertscore=False)
    assert isinstance(result, EvalResult)
    assert result.bertscore_f1 is None
    assert result.n == 1
    out = result.to_dict()
    assert "rouge1_f1" in out
    assert "bertscore_f1" not in out


def test_evaluate_predictions_to_dict_includes_bertscore_when_set() -> None:
    result = EvalResult(n=1, bertscore_f1=0.87)
    out = result.to_dict()
    assert out["bertscore_f1"] == 0.87
