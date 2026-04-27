# Prompt Design

> Status: v1 draft. Iterates with `prompt_version` semver; bumping the
> version invalidates cached labels.

## Goals

1. **Faithful to source.** No new facts, names, or numbers.
2. **Concise.** 2–3 sentences, 40–70 words.
3. **Vietnamese journalistic register.** Neutral, declarative.
4. **Structured output.** JSON with explicit schema for parsing + QC.

## Template

See `configs/prompts/summarize_v1.yaml`. Key sections:
- `system`: instruction text (rules + tone).
- `user_template`: includes title, category, source, and a 6000-char
  truncated `content_text`.
- `response_schema`: JSON schema enforced via Vertex AI
  `response_mime_type: application/json` + `response_schema`.
- `qc`: thresholds consumed by the labeling QC pipeline.

## QC checks (post-generation)

| Check | Definition | Action on fail |
|---|---|---|
| Length | 40–80 words, 2–4 sentences | Drop |
| Entity | Each NER from summary fuzzy-matches (≥ 0.9) into source | Flag |
| Number | Each `\d[\d.,]*` token in summary appears in source | Flag |
| NLI factuality | mDeBERTa-v3 entailment ≥ 0.7 | Flag |
| Refusal | `refusal_reason != null` | Drop |

A label is **accepted into the training set** only if it passes all
checks. Flagged labels are routed to admin review.

## A/B prompt iteration

When testing a new prompt:
1. Bump version (e.g. `1.1.0`) in `configs/prompts/summarize_v1.yaml`
   (or create `summarize_v2.yaml`).
2. Re-label a 200-article holdout set with both versions.
3. Score against gold via ROUGE / BERTScore / human spot-check.
4. Promote the winner. Old labels remain in DB tagged with their version.

## Anti-hallucination patterns

- Explicitly forbid background knowledge ("không bổ sung từ kiến thức ngoài").
- Require numbers/names appear verbatim.
- Low temperature (0.2).
- JSON schema with `confidence` + `refusal_reason` exposes "I don't know".
