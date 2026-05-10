"""Load a fine-tuned ViT5 (+ optional LoRA adapter) for inference.

Supports four layouts under ``model_path``:

1. **Local full model directory** — directly loadable with
   ``AutoModelForSeq2SeqLM`` (i.e. you ran a full fine-tune or merged
   LoRA into the base weights).
2. **Local LoRA adapter directory** — contains an ``adapter_config.json``;
   we load the base model from ``adapter_config.base_model_name_or_path``
   and attach the adapter via PEFT.
3. **HF Hub adapter repo id** — e.g. ``DEFAULT_HF_REPO`` below. We
   download the adapter_config (handles auth via ``HF_TOKEN``) to read
   the base model name, then load ``PeftModel.from_pretrained(base,
   repo_id)``.
4. **Base model name / Hub full-model id** — passing e.g.
   ``VietAI/vit5-base`` skips PEFT entirely and just runs the
   off-the-shelf model (useful as a sanity baseline).
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

DEFAULT_HF_REPO = "Gthgfuiss123/vit5-news-vi-lora-v2"
"""Hub repo holding the v2 LoRA adapter trained on dataset v2.

The repo is private; pass an ``HF_TOKEN`` env var with read access (or
use ``huggingface-cli login``) to load it. See
``docs/training_v2_report.md`` for the full provenance.
"""


@dataclass(slots=True)
class GenerationConfig:
    max_input_length: int = 1024
    max_new_tokens: int = 128
    num_beams: int = 4
    no_repeat_ngram_size: int = 3
    length_penalty: float = 1.0
    early_stopping: bool = True
    batch_size: int = 8
    """Mini-batch size for :meth:`ViT5Summarizer.summarize_batch`.

    Beam search with ``num_beams=4`` over 1024-token inputs blows up VRAM
    quickly, so the eval CLI feeds articles in chunks of this size. 8 is
    safe on a T4 (16 GB); bump it for larger GPUs.
    """


def _is_adapter_dir(path: Path) -> bool:
    return (path / "adapter_config.json").exists()


def _read_base_model_from_adapter(path: Path) -> str:
    cfg = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    base = cfg.get("base_model_name_or_path")
    if not base:
        msg = f"adapter_config.json at {path} is missing base_model_name_or_path"
        raise ValueError(msg)
    return str(base)


def _try_fetch_hub_adapter_config(repo_id: str) -> Path | None:
    """Return a local path to ``adapter_config.json`` if ``repo_id`` is an HF
    Hub adapter repo, else ``None``.

    Used to disambiguate between an adapter-only Hub repo (load via PEFT)
    and a full-model Hub repo (load via ``AutoModelForSeq2SeqLM``).
    Network errors and auth failures are swallowed and surfaced as
    ``None`` so the caller falls back to the full-model branch and gives
    a more useful downstream error.
    """
    try:
        hub = importlib.import_module("huggingface_hub")
    except ImportError:
        return None
    try:
        local = hub.hf_hub_download(repo_id=repo_id, filename="adapter_config.json")
    except Exception as exc:
        logger.debug("adapter_config.json not found on hub for {}: {}", repo_id, exc)
        return None
    return Path(local)


class ViT5Summarizer:
    """Wrapper around a (fine-tuned) ViT5 seq2seq model.

    Heavy ML deps are imported lazily, so unit tests that just exercise
    the constructor / config plumbing don't have to load torch.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        generation: GenerationConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.generation = generation or GenerationConfig()
        self.device = device
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        transformers_mod = importlib.import_module("transformers")
        target = Path(self.model_path)

        if target.is_dir() and _is_adapter_dir(target):
            base_name = _read_base_model_from_adapter(target)
            logger.info("loading base model {} + LoRA adapter at {}", base_name, target)
            base = transformers_mod.AutoModelForSeq2SeqLM.from_pretrained(base_name)
            peft_mod = importlib.import_module("peft")
            self._model = peft_mod.PeftModel.from_pretrained(base, str(target))
            tok_path = str(target if (target / "tokenizer.json").exists() else base_name)
            self._tokenizer = transformers_mod.AutoTokenizer.from_pretrained(tok_path)
        elif not target.exists() and (
            adapter_cfg := _try_fetch_hub_adapter_config(self.model_path)
        ):
            base_name = _read_base_model_from_adapter(adapter_cfg.parent)
            logger.info(
                "loading base model {} + LoRA adapter from hub repo {}",
                base_name,
                self.model_path,
            )
            base = transformers_mod.AutoModelForSeq2SeqLM.from_pretrained(base_name)
            peft_mod = importlib.import_module("peft")
            self._model = peft_mod.PeftModel.from_pretrained(base, self.model_path)
            # Mirror the local-adapter branch: many adapter-only Hub repos
            # ship just the LoRA weights with no tokenizer, so fall back
            # to the base model's tokenizer when the repo doesn't provide
            # one. We attempt the repo first since users can choose to
            # ship a tokenizer alongside the adapter (as we do for v2).
            #
            # ``OSError`` covers "tokenizer files missing"; other exception
            # types can come from format-mismatched tokenizer.json files
            # saved with an older transformers version (we have observed
            # ``TypeError`` here). LoRA never alters the vocab, so the
            # base model tokenizer is always a safe fallback.
            try:
                self._tokenizer = transformers_mod.AutoTokenizer.from_pretrained(self.model_path)
            except (OSError, ValueError, TypeError) as exc:
                logger.info(
                    "tokenizer load from hub repo {} failed ({}); falling back to base model {}",
                    self.model_path,
                    type(exc).__name__,
                    base_name,
                )
                self._tokenizer = transformers_mod.AutoTokenizer.from_pretrained(base_name)
        else:
            logger.info("loading model + tokenizer from {}", self.model_path)
            self._model = transformers_mod.AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            self._tokenizer = transformers_mod.AutoTokenizer.from_pretrained(self.model_path)

        if self.device:
            self._model.to(self.device)
        self._model.eval()
        return self._model, self._tokenizer

    def summarize(self, text: str) -> str:
        """Generate a single Vietnamese summary for ``text``."""
        if not text or not text.strip():
            return ""
        model, tokenizer = self._ensure_loaded()
        inputs = tokenizer(
            text,
            max_length=self.generation.max_input_length,
            truncation=True,
            return_tensors="pt",
        )
        if self.device:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = model.generate(
            **inputs,
            max_new_tokens=self.generation.max_new_tokens,
            num_beams=self.generation.num_beams,
            no_repeat_ngram_size=self.generation.no_repeat_ngram_size,
            length_penalty=self.generation.length_penalty,
            early_stopping=self.generation.early_stopping,
        )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return str(decoded[0]) if decoded else ""

    def summarize_batch(self, texts: list[str], *, batch_size: int | None = None) -> list[str]:
        """Vectorized form of :meth:`summarize`, with mini-batching.

        ``batch_size`` defaults to :attr:`GenerationConfig.batch_size`.
        Inputs are processed in chunks so beam search doesn't try to
        materialize one giant ``(N, max_input_length)`` padded tensor — at
        ``num_beams=4`` and ``max_new_tokens=128`` even modest N would OOM.
        """
        if not texts:
            return []
        # Pre-filter empty/whitespace inputs to match :meth:`summarize`'s
        # contract — those entries always map to "" without invoking the
        # model. Without this, ``summarize_batch != [summarize(t) ...]``
        # for empty inputs.
        empty_mask = [(not t or not t.strip()) for t in texts]
        non_empty = [t for t, is_empty in zip(texts, empty_mask, strict=True) if not is_empty]
        if not non_empty:
            return ["" for _ in texts]

        chunk = batch_size if batch_size and batch_size > 0 else self.generation.batch_size
        chunk = max(chunk, 1)
        model, tokenizer = self._ensure_loaded()

        decoded_iter: list[str] = []
        for i in range(0, len(non_empty), chunk):
            sub = non_empty[i : i + chunk]
            inputs = tokenizer(
                sub,
                max_length=self.generation.max_input_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            if self.device:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.generation.max_new_tokens,
                num_beams=self.generation.num_beams,
                no_repeat_ngram_size=self.generation.no_repeat_ngram_size,
                length_penalty=self.generation.length_penalty,
                early_stopping=self.generation.early_stopping,
            )
            decoded_iter.extend(
                str(s) for s in tokenizer.batch_decode(outputs, skip_special_tokens=True)
            )

        # Re-thread results back into the original order, leaving "" in
        # place for the empty rows.
        out: list[str] = []
        cursor = 0
        for is_empty in empty_mask:
            if is_empty:
                out.append("")
            else:
                out.append(decoded_iter[cursor])
                cursor += 1
        return out
