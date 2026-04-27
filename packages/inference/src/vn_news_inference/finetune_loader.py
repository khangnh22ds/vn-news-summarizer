"""Load a fine-tuned ViT5 (+ optional LoRA adapter) for inference.

Supports three layouts under ``model_path``:

1. **Full model** — directly loadable with ``AutoModelForSeq2SeqLM``
   (i.e. you ran a full fine-tune or merged LoRA into the base weights).
2. **LoRA adapter only** — directory contains an ``adapter_config.json``;
   we load the base model from ``adapter_config.base_model_name_or_path``
   and attach the adapter via PEFT.
3. **Base model name** — passing e.g. ``VietAI/vit5-base`` skips PEFT
   entirely and just runs the off-the-shelf model (useful as a sanity
   baseline against the extractive ones).
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass(slots=True)
class GenerationConfig:
    max_input_length: int = 1024
    max_new_tokens: int = 128
    num_beams: int = 4
    no_repeat_ngram_size: int = 3
    length_penalty: float = 1.0
    early_stopping: bool = True


def _is_adapter_dir(path: Path) -> bool:
    return (path / "adapter_config.json").exists()


def _read_base_model_from_adapter(path: Path) -> str:
    cfg = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    base = cfg.get("base_model_name_or_path")
    if not base:
        msg = f"adapter_config.json at {path} is missing base_model_name_or_path"
        raise ValueError(msg)
    return str(base)


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

    def summarize_batch(self, texts: list[str]) -> list[str]:
        """Vectorized form of :meth:`summarize`."""
        if not texts:
            return []
        model, tokenizer = self._ensure_loaded()
        inputs = tokenizer(
            texts,
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
        return [str(s) for s in tokenizer.batch_decode(outputs, skip_special_tokens=True)]
