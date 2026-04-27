# Training

> Status: outline. Baselines in TICKET-004 (Phase 3); fine-tuning in
> TICKET-005 (Phase 4).

## Models under consideration

| Model | Params | Type | Vietnamese? | Pick? |
|---|---|---|---|---|
| `VietAI/vit5-base` | ~226M | Abstractive (T5) | Trained on Vietnamese | **Default** |
| `VietAI/vit5-large` | ~770M | Abstractive | Vietnamese | If GPU allows |
| `vinai/bartpho-syllable` | ~396M | Abstractive (BART) | Vietnamese | Backup |
| `google/mt5-base` | ~580M | Multilingual | Includes Vietnamese | Diversity |
| `TextRank` (sumy) | — | Extractive | Tokenizer-based | Baseline |

## Compute plan

- **Training:** Colab/Kaggle T4 (free) with **LoRA (r=16, α=32, q+v)** ⇒
  fits in 16GB. 3–5 epochs on ~1500 train pairs ≈ 20–40 min.
- **Inference:** CPU is fine for `vit5-base` (~580MB). Latency ~1–3s/article;
  cached, so user-facing latency is sub-100ms after warm-up.

## Pipeline

1. Load dataset version N from `data/processed/v{N}/{train,val,test}.parquet`.
2. Tokenize with the model's tokenizer (`max_input=1024`, `max_target=128`).
3. `Seq2SeqTrainer` with config from `configs/training/vit5_base_v1.yaml`.
4. Eval each epoch on val: ROUGE-L (best-model selector), ROUGE-1/2,
   BERTScore-F1, NLI factuality.
5. Save best checkpoint to `models/vit5-news-v1/`. Log run to MLflow.

## Evaluation harness

Inputs: model name + dataset version → outputs: a metrics row in MLflow,
plus a JSON report at `data/processed/v{N}/eval_{model}.json`.

Metrics:
- **ROUGE-1 / 2 / L** with pyvi tokenizer.
- **BERTScore-F1** with `xlm-roberta-base`.
- **NLI factuality** (entailment) with mDeBERTa-v3 multilingual.
- **Length conformity** (% in 40–80 words).
- **Human eval** — 50 random items, faithfulness + fluency 1–5.

## Reproducibility

- Pin transformers / torch versions in `packages/training/pyproject.toml`.
- Set `seed=42` in training config.
- Log dataset_version + prompt_version + model commit hash in every
  MLflow run.
- LoRA adapters (small) commit-able if useful; full weights not committed.
