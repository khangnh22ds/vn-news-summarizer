"""Unit tests for the Vertex client wrapper (override path only)."""

from __future__ import annotations

import pytest
from vn_news_labeling import GenerationParams, VertexLabeler, VertexLLMError


def test_override_callable_is_used() -> None:
    seen: dict[str, str] = {}

    def fake(system: str, user: str) -> str:
        seen["system"] = system
        seen["user"] = user
        return '{"summary": "ok", "key_entities": [], "confidence": 0.9}'

    labeler = VertexLabeler(
        project=None,
        location="asia-southeast1",
        model_name="gemini-2.0-flash-001",
        params=GenerationParams(),
        override_callable=fake,
    )
    out = labeler.generate(system="sys", user="usr")
    assert "summary" in out
    assert seen == {"system": "sys", "user": "usr"}


def test_missing_project_raises_when_no_override() -> None:
    labeler = VertexLabeler(
        project=None,
        location="asia-southeast1",
        model_name="gemini-2.0-flash-001",
        params=GenerationParams(),
    )
    with pytest.raises(VertexLLMError, match="GOOGLE_CLOUD_PROJECT"):
        labeler.generate(system="s", user="u")
