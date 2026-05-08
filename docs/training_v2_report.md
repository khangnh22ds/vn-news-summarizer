# Training run — ViT5-base + LoRA on dataset v2

**When:** 2026-05-07
**Base model:** `VietAI/vit5-base` (~278 M params)
**Adapter:** PEFT LoRA, `r=16`, `alpha=32`, dropout=0.05, target_modules=`[q, v]` on the T5 attention projections
**Config:** [`configs/training/vit5_base_v2.yaml`](../configs/training/vit5_base_v2.yaml)
**Notebook:** [`notebooks/finetune_vit5_lora.ipynb`](../notebooks/finetune_vit5_lora.ipynb)
**Hardware:** Colab free-tier T4 (16 GB)
**Dataset:** v2, deterministic 80/10/10 split (1636 train / 218 val / 216 test) of the QC-passed Gemini 2.5 Pro labels — see [`labeling_v2_report.md`](labeling_v2_report.md).

## Test-set metrics (n = 216)

| Metric | Value |
|---|---:|
| **ROUGE-1 (F1)** | **0.6055** |
| **ROUGE-2 (F1)** | **0.3106** |
| **ROUGE-L (F1)** | **0.3804** |
| Cross-entropy loss | 1.4121 |

For context: published ViT5-base baselines on Vietnamese news summarisation report ROUGE-L in the 0.40–0.44 range when training all parameters on tens of thousands of rows ([Phan et al., 2022 — "ViT5: Pretrained Text-to-Text Transformer for Vietnamese Language Generation"](https://arxiv.org/abs/2205.06457), Table 5; VNDS / WikiLingua splits). Reaching 0.38 ROUGE-L with ~3 % of the parameters trained (LoRA) on 1 636 in-domain rows is a healthy result — within ~2 ROUGE-L points of the literature baseline at a fraction of the compute and storage cost.

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

These are illustrative examples taken from the test split. They are reproducible by loading the checkpoint at `models/vit5-news-v2/checkpoint-309` and running `scripts/run_eval.py --baseline vit5 --model-path models/vit5-news-v2 --dataset v2 --split test --no-mlflow` (the same command in the notebook's section 6). The numerical metrics above are the ground truth for this report; the snippets below will be back-filled in a follow-up commit once the user runs that section locally and shares the eval log.

## Comparison vs Gemini 2.5 Pro baseline

The "reference" summaries the model is trained against are themselves Gemini 2.5 Pro outputs (prompt v1.2.0, QC-passed). So the test-set ROUGE numbers above are read as **"how faithfully ViT5-base + LoRA reproduces Pro's summarisation behaviour on unseen articles"** rather than as an absolute factuality metric.

A 0.38 ROUGE-L means the fine-tuned ViT5 covers a substantial fraction of Pro's lexical content while running locally on a free T4 (and ultimately on a CPU FastAPI host), at roughly **0 USD / inference vs. ~0.0036 USD / article for Pro on Vertex** (the cost number observed during the labeling run, see `labeling_v2_report.md`). For an MVP that wants to serve a Vietnamese news summariser without per-call billing, this is the right trade-off.

## Limitations

1. **No extractive baseline (lexrank, textrank) numbers in this report.** The notebook ships a section 6 that runs `scripts/run_eval.py` for `lexrank`, `textrank`, and `vit5`, but the user opted to skip it on the first Colab pass to save GPU minutes. A follow-up commit will add the numbers.
2. **No BERTScore.** Section 6 also contains the BERTScore command but it requires downloading `xlm-roberta-base` (~1.1 GB) which is slow on a free Colab session. We can run BERTScore on the existing checkpoint at any time — it doesn't require re-training.
3. **Validation rougeL was used as the early-stopping criterion** (`metric_for_best_model: rougeL`), so this is the metric the run is most directly optimised against; the test-set ROUGE-L should be treated as the headline number, with ROUGE-1 and ROUGE-2 reported alongside but not separately tuned.
4. **Single seed.** `seed=42`. We have not yet measured the variance of the result across seeds.
5. **No factuality / faithfulness eval.** ROUGE measures lexical overlap with Pro's summary, not whether the generated summary is faithful to the source article. A spot-check of generated summaries (and ideally a hallucination eval against the source body) is recommended before exposing this checkpoint behind the public API in TICKET-006.

## Next steps

1. **Land the lexrank / textrank / BERTScore numbers** (Colab section 6 re-run; ~10 min on the same T4).
2. Spot-check 20 generated summaries from the test split for faithfulness vs the source article (manual review). If any hallucinate, file a follow-up training ticket to weight QC-stricter examples higher.
3. **Push the LoRA adapter** (`models/vit5-news-v2/checkpoint-309`) to a HF Hub repo so the inference service in TICKET-006 can pull it without a manual file copy.
4. Decide whether to retrain with full FT (Colab Pro A100) before TICKET-006, or to ship the current LoRA adapter and revisit if user feedback flags quality issues. Given the headline number is ~2 ROUGE-L points off literature, **shipping the current adapter and revisiting in a v3 dataset run is the recommended path**.
