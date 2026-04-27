"""Evaluation harness for summarization baselines.

Computes ROUGE-1 / ROUGE-2 / ROUGE-L F1 (and optionally BERTScore F1)
against a list of (system, reference) pairs.

BERTScore is expensive (loads a transformer + downloads weights on first
call) and not available in CI without network egress, so it's gated
behind an explicit flag and only invoked when the caller asks for it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from rouge_score import rouge_scorer


@dataclass(slots=True)
class RougeAggregate:
    """Mean ROUGE F1 scores across a corpus."""

    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    n: int = 0


@dataclass(slots=True)
class EvalResult:
    """All metrics for a single (baseline, dataset, split) run."""

    rouge: RougeAggregate = field(default_factory=RougeAggregate)
    bertscore_f1: float | None = None
    n: int = 0

    def to_dict(self) -> dict[str, float | int]:
        out: dict[str, float | int] = {
            "rouge1_f1": round(self.rouge.rouge1, 4),
            "rouge2_f1": round(self.rouge.rouge2, 4),
            "rougeL_f1": round(self.rouge.rougeL, 4),
            "n": self.n,
        }
        if self.bertscore_f1 is not None:
            out["bertscore_f1"] = round(self.bertscore_f1, 4)
        return out


def compute_rouge(
    predictions: Sequence[str],
    references: Sequence[str],
) -> RougeAggregate:
    """Mean ROUGE-1/2/L F1 over a parallel list of strings.

    Empty inputs return zeroed scores.
    """
    if not predictions:
        return RougeAggregate()
    if len(predictions) != len(references):
        msg = f"predictions ({len(predictions)}) and references ({len(references)}) length mismatch"
        raise ValueError(msg)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    r1 = r2 = rL = 0.0
    for pred, ref in zip(predictions, references, strict=False):
        score = scorer.score(ref, pred)
        r1 += score["rouge1"].fmeasure
        r2 += score["rouge2"].fmeasure
        rL += score["rougeL"].fmeasure
    n = len(predictions)
    return RougeAggregate(
        rouge1=r1 / n,
        rouge2=r2 / n,
        rougeL=rL / n,
        n=n,
    )


def compute_bertscore(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    model_type: str = "xlm-roberta-base",
    lang: str = "vi",
    batch_size: int = 8,
) -> float:
    """Mean BERTScore F1 over a parallel list of strings.

    Lazy-imports ``bert_score`` so the (heavy) torch/transformers stack
    isn't loaded for unit tests that don't ask for it. Network access
    is required on first call to download the chosen model.
    """
    if not predictions:
        return 0.0
    if len(predictions) != len(references):
        msg = "predictions and references length mismatch"
        raise ValueError(msg)
    from bert_score import score

    _, _, f1 = score(
        list(predictions),
        list(references),
        model_type=model_type,
        lang=lang,
        batch_size=batch_size,
        verbose=False,
    )
    return float(f1.mean().item())


def evaluate_predictions(
    predictions: Iterable[str],
    references: Iterable[str],
    *,
    use_bertscore: bool = False,
    bertscore_kwargs: dict[str, object] | None = None,
) -> EvalResult:
    """Run all enabled metrics and bundle the results."""
    preds = list(predictions)
    refs = list(references)
    rouge = compute_rouge(preds, refs)
    bs: float | None = None
    if use_bertscore:
        kwargs = bertscore_kwargs or {}
        bs = compute_bertscore(preds, refs, **kwargs)  # type: ignore[arg-type]
    return EvalResult(rouge=rouge, bertscore_f1=bs, n=len(preds))
