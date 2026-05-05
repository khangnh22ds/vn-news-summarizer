"""Prompt template loader and renderer for LLM labeling.

Reads ``configs/prompts/summarize_v1.yaml`` and exposes a :class:`Prompt`
that can be rendered against an article. Outputs a JSON-mode-friendly
structure expected by Gemini ``response_mime_type=application/json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError


class GenerationCfg(BaseModel):
    """LLM generation parameters (passed verbatim to Vertex)."""

    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    # Gemini 2.5 models consume output budget for thinking tokens + response;
    # 2048 leaves enough headroom for long Vietnamese articles without OOM.
    max_output_tokens: int = Field(default=2048, ge=16, le=8192)
    response_mime_type: str = Field(default="application/json")


class QcCfg(BaseModel):
    """QC thresholds. NLI fields are kept for future tickets but unused here."""

    min_words: int = Field(default=40, ge=1)
    max_words: int = Field(default=80, ge=1)
    min_sentences: int = Field(default=2, ge=1)
    max_sentences: int = Field(default=4, ge=1)
    entity_fuzzy_min_ratio: float = Field(default=0.9, ge=0.0, le=1.0)
    nli_entailment_min: float = Field(default=0.7, ge=0.0, le=1.0)
    nli_model: str | None = Field(default=None)


@dataclass(slots=True)
class Prompt:
    """A loaded prompt template ready to render against an article."""

    version: str
    model: str
    provider: str
    system: str
    user_template: str
    generation: GenerationCfg
    qc: QcCfg
    response_schema: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Prompt:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        try:
            generation = GenerationCfg(**(raw.get("generation_config") or {}))
            qc = QcCfg(**(raw.get("qc") or {}))
        except ValidationError as exc:  # pragma: no cover — config bug
            msg = f"invalid prompt config {path}: {exc}"
            raise ValueError(msg) from exc
        return cls(
            version=str(raw.get("version", "0.0.0")),
            model=str(raw.get("model", "gemini-2.5-flash")),
            provider=str(raw.get("provider", "vertex_ai")),
            system=str(raw.get("system", "")).strip(),
            user_template=str(raw.get("user_template", "")).strip(),
            generation=generation,
            qc=qc,
            response_schema=dict(raw.get("response_schema") or {}),
        )

    def render_user(
        self,
        *,
        title: str,
        category: str | None,
        source: str,
        content_text: str,
        content_max_chars: int = 6000,
    ) -> str:
        """Render the user message, truncating ``content_text``."""
        snippet = content_text or ""
        if len(snippet) > content_max_chars:
            snippet = snippet[:content_max_chars].rsplit(" ", 1)[0] + " […]"
        return self.user_template.format(
            title=title or "",
            category=category or "",
            source=source,
            content_text=snippet,
        )


class LabelOutput(BaseModel):
    """Structured response from the LLM (after JSON parsing)."""

    summary: str
    key_entities: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    refusal_reason: str | None = None


def parse_label_json(raw_text: str) -> LabelOutput:
    """Parse the JSON-mode response into :class:`LabelOutput`.

    Raises ``ValueError`` if the JSON is malformed or doesn't match the
    expected shape.
    """
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        msg = f"LLM did not return valid JSON: {exc}"
        raise ValueError(msg) from exc
    try:
        return LabelOutput(**data)
    except ValidationError as exc:
        msg = f"LLM JSON does not match schema: {exc}"
        raise ValueError(msg) from exc
