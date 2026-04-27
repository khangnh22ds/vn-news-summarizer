# Training

> Status: TICKET-004 (extractive baselines) ✅ ; TICKET-005 (ViT5 + LoRA
> fine-tune) ✅ ; future tickets layer NLI factuality and human eval on
> top.

## Models under consideration

| Model | Params | Type | Vietnamese? | Pick? |
|---|---|---|---|---|
| `VietAI/vit5-base` | ~226M | Abstractive (T5) | Trained on Vietnamese | **Default** |
| `VietAI/vit5-large` | ~770M | Abstractive | Vietnamese | If GPU allows |
| `vinai/bartpho-syllable` | ~396M | Abstractive (BART) | Vietnamese | Backup |
| `google/mt5-base` | ~580M | Multilingual | Includes Vietnamese | Diversity |
| `LexRank` / `TextRank` (in-house) | — | Extractive | Tokenizer-based | Baseline |

## Compute plan

- **Training:** Colab/Kaggle T4 (free) with **LoRA (r=16, α=32, q+v)** ⇒
  fits in 16 GB. 3–5 epochs on ~1 500 train pairs ≈ 20–40 min.
- **Inference:** CPU is fine for `vit5-base` (~580 MB). Latency ~1–3 s/article;
  cached, so user-facing latency is sub-100 ms after warm-up.

## Pipeline

1. Build a labeled dataset version with `make label-export NAME=v1` —
   writes `data/datasets/v1/{train,val,test}.jsonl`.
2. `vn_news_training.preprocess.build_hf_dataset_dict` tokenizes each
   split with the model's tokenizer (`max_input=1024`, `max_target=128`).
3. `Seq2SeqTrainer` (transformers ≥ 4.46) with config from
   `configs/training/vit5_base_v1.yaml`. PEFT `LoraConfig` wraps the
   base model when `peft.enabled: true`.
4. Eval each epoch on val: ROUGE-L (best-model selector), ROUGE-1/2.
5. Save the best checkpoint to `models/vit5-news-v1/`; the LoRA adapter
   weights live next to a `tokenizer.json`. MLflow gets a run with
   metrics + params.

## Run on Colab / Kaggle

The fastest path is the companion notebook: open
`notebooks/finetune_vit5_lora.ipynb` in Colab, mount Drive, copy the
JSONL splits over, run the "Run the fine-tune" cell, and tar the output
back to Drive when you're done.

The CLI version is the same code:

```bash
# Defaults (configs/training/vit5_base_v1.yaml)
make train

# Override config / dataset / output
make train CONFIG=configs/training/vit5_base_v1.yaml DATASET=v1 \
            EPOCHS=3 OUTPUT=models/vit5-news-v1
```

## Evaluating a fine-tuned checkpoint

`scripts/run_eval.py` knows how to load both the extractive baselines
and a fine-tuned ViT5 / LoRA checkpoint:

```bash
# Extractive
make eval BASELINE=lexrank DATASET=v1 SPLIT=test
make eval BASELINE=textrank DATASET=v1 SPLIT=test

# Fine-tuned model
make eval BASELINE=vit5 MODEL=models/vit5-news-v1 DATASET=v1 SPLIT=test
```

`MODEL=` may also be a HuggingFace base model name (e.g.
`VietAI/vit5-base`) for a zero-shot sanity baseline.

## Evaluation harness

Inputs: model + dataset version → outputs: a metrics dict in MLflow.

Metrics shipped today:

- **ROUGE-1 / 2 / L** F1 (regex tokenizer; underthesea is opt-in via
  `VN_NEWS_USE_UNDERTHESEA=1`).
- **BERTScore-F1** with `xlm-roberta-base` (lazy import; opt-in via
  `--bertscore`).

Planned (not yet shipped):

- NLI factuality with mDeBERTa-v3 multilingual.
- Length conformity (% in 40–80 words).
- Human eval — 50 random items, faithfulness + fluency 1–5.

## Reproducibility

- Pin transformers / torch versions in `packages/training/pyproject.toml`.
- Set `seed=42` in training config.
- Log dataset_version + prompt_version + model commit hash in every
  MLflow run.
- LoRA adapters are tiny (~10 MB) so they're easy to ship; full base
  weights are not committed.
