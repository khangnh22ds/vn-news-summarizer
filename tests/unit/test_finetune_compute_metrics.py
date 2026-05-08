"""Unit tests for the seq2seq compute_metrics closure.

Pins the fix for the OverflowError observed on Colab T4 at the end of
the first training epoch: when ``predict_with_generate=True``,
transformers pads ragged generation outputs in ``all_preds`` with
``-100``. The fast tokenizer's Rust decoder casts each id to ``u32``
and raises ``OverflowError: out of range integral type conversion
attempted`` on negative values. The closure must therefore replace any
negative id with the tokenizer's pad id *before* calling
``batch_decode`` — for **both** the predictions array and the labels
array.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from vn_news_training.finetune import _make_compute_metrics


class _FakeFastTokenizer:
    """Mimic the failure mode of a HF fast tokenizer.

    The real ``PreTrainedTokenizerFast.batch_decode`` calls Rust code
    that does ``u32::try_from(id)`` per token; we approximate that here
    by raising ``OverflowError`` on any negative id, exactly like the
    crash the user saw on Colab.
    """

    pad_token_id: int = 0

    def batch_decode(
        self,
        sequences: Any,
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        del skip_special_tokens
        seqs = np.asarray(sequences)
        if (seqs < 0).any():
            raise OverflowError("out of range integral type conversion attempted")
        # Concatenate ids into a fake "decoded" string so we can also
        # verify the closure feeds the right ids through.
        return [" ".join(str(int(t)) for t in row) for row in seqs]


def test_compute_metrics_replaces_negative_ids_in_preds_and_labels() -> None:
    """The exact crash from Colab: preds contain -100, labels contain -100.

    Without the fix the closure would let ``batch_decode`` see the
    ``-100`` ids and raise ``OverflowError``. With the fix both arrays
    are sanitised to ``pad_token_id`` first.
    """
    tokenizer = _FakeFastTokenizer()
    compute = _make_compute_metrics(tokenizer)

    # 2 examples; predictions ragged -> padded with -100 by transformers.
    preds = np.array(
        [
            [10, 20, 30, -100],
            [40, 50, -100, -100],
        ]
    )
    labels = np.array(
        [
            [10, 20, 30, -100],
            [40, 50, 60, -100],
        ]
    )

    metrics = compute((preds, labels))

    # Primary contract: the closure ran to completion without raising
    # OverflowError, even though preds/labels both contained -100. The
    # exact ROUGE number depends on the fake decoder, but it must be
    # finite and within [0, 1].
    assert set(metrics) == {"rouge1", "rouge2", "rougeL"}
    for k in ("rouge1", "rouge2", "rougeL"):
        assert 0.0 <= metrics[k] <= 1.0
    # Sanity: the first example is a perfect match after sanitisation
    # and the second is a 3/4 unigram match, so ROUGE-1 should be
    # comfortably above zero.
    assert metrics["rouge1"] > 0.5


def test_compute_metrics_unwraps_tuple_predictions() -> None:
    """Some HF versions return preds as a (sequences, scores) tuple."""
    tokenizer = _FakeFastTokenizer()
    compute = _make_compute_metrics(tokenizer)

    preds_seq = np.array([[1, 2, 3]])
    extra = np.array([[0.1, 0.2, 0.3]])  # scores; ignored.
    labels = np.array([[1, 2, 3]])

    metrics = compute(((preds_seq, extra), labels))
    assert metrics["rouge1"] == 1.0


def test_compute_metrics_handles_all_negative_predictions() -> None:
    """Edge case: every prediction is -100 (e.g. very short inputs).

    The closure must not crash and must report a finite ROUGE score
    (likely 0.0) instead of raising.
    """
    tokenizer = _FakeFastTokenizer()
    compute = _make_compute_metrics(tokenizer)

    preds = np.full((2, 4), -100, dtype=np.int64)
    labels = np.array([[10, 20, 30, 40], [50, 60, -100, -100]])

    metrics = compute((preds, labels))
    assert 0.0 <= metrics["rouge1"] <= 1.0
