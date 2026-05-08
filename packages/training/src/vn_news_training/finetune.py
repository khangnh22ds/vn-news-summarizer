"""Fine-tuning orchestrator for ViT5 + LoRA on Vietnamese news summaries.

This module is intentionally thin: it composes the existing pieces
(``preprocess.build_hf_dataset_dict``, transformers' ``Seq2SeqTrainer``,
PEFT's ``get_peft_model``) and exposes a single :func:`run_finetune`
entrypoint. The heavy ML libraries are lazy-imported so importing the
package on a CPU-only CI runner without GPU/transformers fully cached
is still fast.

Typical flow on a Colab T4 with ``configs/training/vit5_base_v1.yaml``
takes ~30 min for ~1 000 articles, ~3 epochs.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from .config import FinetuneConfig, load_finetune_config
from .preprocess import build_hf_dataset_dict


@dataclass(slots=True)
class TrainResult:
    """What :func:`run_finetune` returns."""

    output_dir: Path
    metrics: dict[str, float]
    best_model_checkpoint: str | None = None


def _build_lora_config(cfg: FinetuneConfig) -> Any:
    """Build a PEFT ``LoraConfig`` for seq2seq language modeling."""
    peft_mod = importlib.import_module("peft")
    return peft_mod.LoraConfig(
        r=cfg.peft.r,
        lora_alpha=cfg.peft.alpha,
        lora_dropout=cfg.peft.dropout,
        bias="none",
        task_type=peft_mod.TaskType.SEQ_2_SEQ_LM,
        target_modules=list(cfg.peft.target_modules),
    )


def _build_training_arguments(cfg: FinetuneConfig) -> Any:
    """Build a ``Seq2SeqTrainingArguments`` from the YAML config."""
    transformers_mod = importlib.import_module("transformers")
    t = cfg.training
    return transformers_mod.Seq2SeqTrainingArguments(
        output_dir=t.output_dir,
        num_train_epochs=t.num_train_epochs,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        warmup_ratio=t.warmup_ratio,
        weight_decay=t.weight_decay,
        fp16=t.fp16,
        bf16=t.bf16,
        eval_strategy=t.eval_strategy,
        save_strategy=t.save_strategy,
        load_best_model_at_end=t.load_best_model_at_end,
        metric_for_best_model=t.metric_for_best_model,
        greater_is_better=t.greater_is_better,
        predict_with_generate=t.predict_with_generate,
        generation_num_beams=t.generation_num_beams,
        generation_max_length=t.generation_max_length,
        logging_steps=t.logging_steps,
        save_total_limit=t.save_total_limit,
        seed=t.seed,
        report_to=list(t.report_to),
        run_name=cfg.run_name,
    )


def _make_compute_metrics(
    tokenizer: Any,
) -> Any:
    """Closure that decodes generated ids and computes ROUGE-1/2/L."""
    rouge_mod = importlib.import_module("rouge_score.rouge_scorer")
    scorer = rouge_mod.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

    def _compute(eval_pred: Any) -> dict[str, float]:
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        # transformers fills label-pad with -100, and pads ragged
        # generation outputs in ``all_preds`` with -100 too. The fast
        # tokenizer's Rust decoder casts each id to ``u32`` and raises
        # ``OverflowError`` on negatives, so swap ANY negative value
        # back to the tokenizer's pad id before decoding (preds + labels).
        pad_id = tokenizer.pad_token_id
        preds = np.asarray(preds)
        labels = np.asarray(labels)
        preds = np.where(preds < 0, pad_id, preds)
        labels = np.where(labels < 0, pad_id, labels)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        r1 = r2 = rL = 0.0
        n = len(decoded_preds)
        for pred, ref in zip(decoded_preds, decoded_labels, strict=False):
            sc = scorer.score(ref, pred)
            r1 += sc["rouge1"].fmeasure
            r2 += sc["rouge2"].fmeasure
            rL += sc["rougeL"].fmeasure
        if n == 0:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        return {
            "rouge1": r1 / n,
            "rouge2": r2 / n,
            "rougeL": rL / n,
        }

    return _compute


def run_finetune(
    config_path: Path | str,
    *,
    cfg: FinetuneConfig | None = None,
) -> TrainResult:
    """Run the full fine-tune loop end to end.

    Either ``config_path`` (read+parse YAML) or a pre-built ``cfg`` may
    be supplied; passing ``cfg`` is mostly for tests.
    """
    cfg = cfg or load_finetune_config(config_path)
    transformers_mod = importlib.import_module("transformers")
    peft_mod = importlib.import_module("peft")

    logger.info("loading tokenizer + base model: {}", cfg.model.base)
    tokenizer = transformers_mod.AutoTokenizer.from_pretrained(cfg.model.base)
    base_model = transformers_mod.AutoModelForSeq2SeqLM.from_pretrained(cfg.model.base)

    if cfg.peft.enabled:
        lora_cfg = _build_lora_config(cfg)
        model = peft_mod.get_peft_model(base_model, lora_cfg)
        model.print_trainable_parameters()
    else:
        model = base_model

    logger.info("tokenizing splits from {}/{}", cfg.dataset.root, cfg.dataset.name)
    ds_dict = build_hf_dataset_dict(
        dataset_cfg=cfg.dataset,
        model_cfg=cfg.model,
        tokenizer=tokenizer,
    )

    data_collator = transformers_mod.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        label_pad_token_id=-100,
    )

    args = _build_training_arguments(cfg)
    trainer = transformers_mod.Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds_dict.get("train"),
        eval_dataset=ds_dict.get("val"),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_make_compute_metrics(tokenizer),
    )

    logger.info(
        "starting training: epochs={} steps_per_epoch~={}",
        cfg.training.num_train_epochs,
        len(ds_dict.get("train", [])) // max(cfg.training.per_device_train_batch_size, 1),
    )
    train_out = trainer.train()
    metrics: dict[str, float] = dict(train_out.metrics or {})

    if "test" in ds_dict and len(ds_dict["test"]) > 0:
        eval_metrics = trainer.evaluate(eval_dataset=ds_dict["test"], metric_key_prefix="test")
        metrics.update({k: float(v) for k, v in eval_metrics.items() if isinstance(v, int | float)})

    output_dir = Path(cfg.training.output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    return TrainResult(
        output_dir=output_dir,
        metrics={k: float(v) for k, v in metrics.items() if isinstance(v, int | float)},
        best_model_checkpoint=getattr(trainer.state, "best_model_checkpoint", None),
    )
