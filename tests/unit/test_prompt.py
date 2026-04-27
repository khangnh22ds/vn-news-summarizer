"""Unit tests for the prompt loader and JSON parser."""

from __future__ import annotations

from pathlib import Path

import pytest
from vn_news_labeling import LabelOutput, Prompt, parse_label_json

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = REPO_ROOT / "configs" / "prompts" / "summarize_v1.yaml"


def test_prompt_loads_from_repo_yaml() -> None:
    p = Prompt.from_yaml(PROMPT_PATH)
    assert p.version == "1.0.0"
    assert p.provider == "vertex_ai"
    assert "biên tập viên" in p.system
    assert "{title}" in p.user_template
    assert p.generation.response_mime_type == "application/json"
    assert p.qc.min_words >= 1
    assert p.qc.max_words >= p.qc.min_words


def test_prompt_render_truncates_long_content() -> None:
    p = Prompt.from_yaml(PROMPT_PATH)
    long_text = "Việt Nam " * 2000  # ~16000 chars
    rendered = p.render_user(
        title="Tiêu đề",
        category="thoi-su",
        source="VnExpress",
        content_text=long_text,
        content_max_chars=500,
    )
    assert "Tiêu đề" in rendered
    assert "VnExpress" in rendered
    # Content was truncated.
    assert "[…]" in rendered
    # Rendered length is bounded (template + ~500 chars).
    assert len(rendered) < 1500


def test_parse_label_json_happy_path() -> None:
    raw = """{
        "summary": "Hôm nay TP HCM ghi nhận 12 ca mắc mới.",
        "key_entities": ["TP HCM"],
        "confidence": 0.92,
        "refusal_reason": null
    }"""
    out = parse_label_json(raw)
    assert isinstance(out, LabelOutput)
    assert out.summary.startswith("Hôm nay")
    assert out.confidence == pytest.approx(0.92)
    assert out.key_entities == ["TP HCM"]
    assert out.refusal_reason is None


def test_parse_label_json_missing_field() -> None:
    raw = '{"summary": "x", "confidence": 0.5}'
    # Missing required key_entities is OK because schema gives default;
    # but malformed JSON -> ValueError.
    out = parse_label_json(raw)
    assert out.key_entities == []


def test_parse_label_json_invalid_json() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        parse_label_json("not-json")


def test_parse_label_json_invalid_schema() -> None:
    raw = '{"summary": "x", "confidence": 5.0}'  # confidence > 1
    with pytest.raises(ValueError, match="schema"):
        parse_label_json(raw)
