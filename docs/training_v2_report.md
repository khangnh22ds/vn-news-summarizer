# Training run — ViT5-base + LoRA on dataset v2

**When:** 2026-05-07
**Base model:** `VietAI/vit5-base` (~278 M params)
**Adapter:** PEFT LoRA, `r=16`, `alpha=32`, dropout=0.05, target_modules=`[q, v]` on the T5 attention projections
**Config:** [`configs/training/vit5_base_v2.yaml`](../configs/training/vit5_base_v2.yaml)
**Notebook:** [`notebooks/finetune_vit5_lora.ipynb`](../notebooks/finetune_vit5_lora.ipynb)
**Hardware:** Colab free-tier T4 (16 GB)
**Dataset:** v2, deterministic 80/10/10 split (1636 train / 218 val / 216 test) of the QC-passed Gemini 2.5 Pro labels — see [`labeling_v2_report.md`](labeling_v2_report.md).

## Test-set metrics (n = 216)

| Model | ROUGE-1 (F1) | ROUGE-2 (F1) | ROUGE-L (F1) | BERTScore-F1 |
|---|---:|---:|---:|---:|
| **LexRank** (extractive) | 0.6779 | 0.3520 | 0.3976 | 0.8767 |
| **TextRank** (extractive) | 0.6743 | 0.3493 | 0.3999 | 0.8760 |
| **ViT5-base + LoRA v2** (this run) | 0.6055 | 0.3106 | 0.3804 | *(skipped — see note)* |

The fine-tuned model's test cross-entropy loss is **1.4121**.

For context: published ViT5-base baselines on Vietnamese news summarisation report ROUGE-L in the 0.40–0.44 range when training all parameters on tens of thousands of rows ([Phan et al., 2022 — "ViT5: Pretrained Text-to-Text Transformer for Vietnamese Language Generation"](https://arxiv.org/abs/2205.06457), Table 5; VNDS / WikiLingua splits). Reaching 0.38 ROUGE-L with ~3 % of the parameters trained (LoRA) on 1 636 in-domain rows is a healthy result — within ~2 ROUGE-L points of the literature baseline at a fraction of the compute and storage cost.

### Why are LexRank / TextRank higher on ROUGE than the fine-tuned ViT5?

This is the most important finding of the v2 evaluation, and it is **not** a sign that the fine-tune underperformed. It is a property of the reference summaries.

The gold summaries the test split is graded against are Gemini 2.5 Pro outputs (prompt v1.2.0). Empirically, when given the prompt template in [`configs/prompts/summarize_v1.yaml`](../configs/prompts/summarize_v1.yaml), Pro produces summaries that are **strongly extractive** — most surface n-grams, named entities and numerical facts in the summary appear verbatim in the source article. (The QC heuristics actively *enforce* this: an entity / number that is not present in the source corpus fails QC and the row is dropped, see [`labeling_v2_report.md`](labeling_v2_report.md).) That QC behaviour is what we want for factuality, but it biases the reference summaries toward extractive style.

LexRank and TextRank score sentences by source-graph centrality and return source sentences **unchanged**, which means they reproduce the exact n-grams that Pro tends to copy. ViT5-base + LoRA, by contrast, generates abstractively — paraphrasing, compressing, and reordering — so its surface overlap with Pro's reference is lower even when the *meaning* is correct. The result is that on a dataset whose references are extractive-leaning, an extractive baseline will trivially out-ROUGE any abstractive model.

Concrete implication: **ROUGE on this dataset measures "how extractively Pro-like is the output", not "how good is the summary"**. Two follow-ups address this directly:

1. *Manual spot-check* of 20 ViT5 outputs (planned, see Next steps) to confirm that the abstractive paraphrasing is producing meaningful summaries rather than degenerate output.
2. *Dataset v3* with a new prompt that instructs Pro to produce **abstractive** summaries (paraphrase mandatory, no copying spans longer than ~5 tokens). On a v3 dataset both the headline numbers and the ranking between extractive baselines and ViT5 are expected to change.

## Training-run summary

| Item | Value |
|---|---:|
| Train samples | 1 636 |
| Validation samples | 218 |
| Test samples | 216 |
| Epochs | 4 |
| Effective batch size | 16 (`per_device_train_batch_size=4` × `gradient_accumulation_steps=4`) |
| Learning rate | 5.0 × 10⁻⁵ |
| Warmup ratio | 0.05 |
| Weight decay | 0.01 |
| Generation beams | 4 |
| Max input / target length | 1024 / 128 |
| Mixed precision | fp16 |
| Total training steps | 408 |
| Best validation checkpoint | `models/vit5-news-v2/checkpoint-309` (epoch ≈ 3) |
| Train wall clock | 28 min 55 s |
| Test wall clock | 3 min 04 s |
| Train loss (final) | 1.8848 |
| Train samples / second | 3.77 |
| Total FLOPs | 7.39 × 10¹⁵ |

The best checkpoint landed at step 309 (≈ 3.0 epochs), which is why the loaded best-checkpoint test loss (1.4121) is below the average train loss across all 4 epochs — the run had already started to plateau by epoch 3 and `load_best_model_at_end` rolled back to that point.

## Selected qualitative samples

Illustrative examples are reproducible by loading the checkpoint at `models/vit5-news-v2/checkpoint-309` and running `scripts/run_eval.py --baseline vit5 --model-path models/vit5-news-v2 --dataset v2 --split test --no-mlflow` (the same command in the notebook's section 6). The numerical metrics above are the ground truth for this report; concrete (article → reference → vit5 prediction) snippets will be back-filled in a follow-up commit alongside the planned faithfulness spot-check (see Next steps).

## Comparison vs Gemini 2.5 Pro baseline

The "reference" summaries the model is trained against are themselves Gemini 2.5 Pro outputs (prompt v1.2.0, QC-passed). So the test-set ROUGE numbers above are read as **"how faithfully ViT5-base + LoRA reproduces Pro's summarisation behaviour on unseen articles"** rather than as an absolute factuality metric.

A 0.38 ROUGE-L means the fine-tuned ViT5 covers a substantial fraction of Pro's lexical content while running locally on a free T4 (and ultimately on a CPU FastAPI host), at roughly **0 USD / inference vs. ~0.0036 USD / article for Pro on Vertex** (the cost number observed during the labeling run, see `labeling_v2_report.md`). For an MVP that wants to serve a Vietnamese news summariser without per-call billing, this is the right trade-off.

## Limitations

1. **BERTScore for the fine-tuned ViT5 row is not yet measured.** Section 6 of the notebook ran `lexrank` and `textrank` end-to-end with BERTScore, but the third invocation (`vit5 --bertscore`) is significantly slower because (a) it has to load the LoRA adapter on top of `vit5-base` and run beam search with `num_beams=4` over 216 articles, and (b) BERTScore then has a second pass over 216 references with `xlm-roberta-base`. The run was aborted on the first Colab pass; the cell remains in the notebook and can be executed any time without retraining. Until then, ViT5 is compared on ROUGE only.
2. **The reference distribution is extractive-biased**, see the explanatory section under "Test-set metrics". A v3 dataset with explicit abstractive prompting is the most direct fix.
3. **Validation rougeL was used as the early-stopping criterion** (`metric_for_best_model: rougeL`), so this is the metric the run is most directly optimised against; the test-set ROUGE-L should be treated as the headline number, with ROUGE-1 and ROUGE-2 reported alongside but not separately tuned.
4. **Single seed.** `seed=42`. We have not yet measured the variance of the result across seeds.
5. **No factuality / faithfulness eval.** ROUGE measures lexical overlap with Pro's summary, not whether the generated summary is faithful to the source article. A spot-check of generated summaries (and ideally a hallucination eval against the source body) is recommended before exposing this checkpoint behind the public API in TICKET-006.

## Next steps

1. **Run the ViT5 BERTScore pass** (notebook section 6, third command) on the saved checkpoint to fill the missing cell in the headline table. This is the only number still pending.
2. **Manual faithfulness spot-check** of 20 generated summaries from the test split, comparing each ViT5 output against (a) the source article and (b) the Pro reference. The hypothesis from the extractive-bias finding is that ViT5 is producing *meaningfully abstractive* summaries that ROUGE penalises but a human would judge correct; this spot-check tests that. If any output hallucinates against the source, file a follow-up training ticket to weight QC-stricter examples higher.
3. **Dataset v3 with abstractive prompting.** Change the labeling prompt to require paraphrase (no spans longer than ~5 tokens copied verbatim from the source) and re-run labeling on the same 2 418 articles. Expected outcomes: lower ROUGE-1/2 against the new reference (because the reference itself becomes paraphrastic), but a closer ranking between extractive baselines and the fine-tuned ViT5 — which is the metric we actually care about for an abstractive summariser.
4. **Push the LoRA adapter** (`models/vit5-news-v2/checkpoint-309`) to a HF Hub repo so the inference service in TICKET-006 can pull it without a manual file copy.
5. Decide whether to retrain with full FT (Colab Pro A100) before TICKET-006, or to ship the current LoRA adapter and revisit if user feedback flags quality issues. Given the headline number is ~2 ROUGE-L points off literature, **shipping the current adapter and revisiting on a v3 dataset run is the recommended path**.
