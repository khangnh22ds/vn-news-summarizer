"""Unit tests for the QC battery (entity + number + length checks)."""

from __future__ import annotations

from vn_news_labeling import LabelOutput, run_qc
from vn_news_labeling.prompt import QcCfg


def _cfg(**overrides: float) -> QcCfg:
    base: dict[str, float] = {
        "min_words": 10,
        "max_words": 30,
        "min_sentences": 1,
        "max_sentences": 5,
        "entity_fuzzy_min_ratio": 0.9,
    }
    base.update(overrides)
    return QcCfg(**base)  # type: ignore[arg-type]


SOURCE = (
    "Hôm 12/04, Bộ Y tế cho biết TP HCM ghi nhận 1.234 ca mắc Covid-19 mới. "
    "Chủ tịch Nguyễn Văn A đã chỉ đạo triển khai 5 biện pháp cấp bách "
    "để phòng dịch tại các quận trung tâm."
)


def test_qc_passes_for_faithful_summary() -> None:
    out = LabelOutput(
        summary=(
            "Bộ Y tế cho biết TP HCM ghi nhận 1.234 ca Covid-19 mới ngày 12/04. "
            "Chủ tịch Nguyễn Văn A chỉ đạo triển khai 5 biện pháp cấp bách."
        ),
        key_entities=["TP HCM", "Bộ Y tế"],
        confidence=0.9,
    )
    res = run_qc(output=out, source_text=SOURCE, cfg=_cfg())
    assert res.passed, res.reasons
    assert res.word_count >= 10
    assert not res.missing_numbers
    assert not res.missing_entities


def test_qc_flags_hallucinated_number() -> None:
    out = LabelOutput(
        summary=(
            "Bộ Y tế cho biết TP HCM ghi nhận 9.999 ca Covid-19 mới ngày 12/04. "
            "Chủ tịch Nguyễn Văn A chỉ đạo các biện pháp cấp bách."
        ),
        key_entities=["TP HCM"],
        confidence=0.9,
    )
    res = run_qc(output=out, source_text=SOURCE, cfg=_cfg())
    assert not res.passed
    assert any("unsupported_numbers" in r for r in res.reasons)
    assert "9.999" in res.missing_numbers


def test_qc_flags_hallucinated_entity() -> None:
    out = LabelOutput(
        summary=(
            "Bộ Y tế cho biết Hà Nội ghi nhận 1.234 ca Covid-19 mới ngày 12/04. "
            "Chủ tịch Trần Văn B chỉ đạo các biện pháp cấp bách."
        ),
        key_entities=["Hà Nội"],
        confidence=0.9,
    )
    res = run_qc(output=out, source_text=SOURCE, cfg=_cfg())
    assert not res.passed
    assert any("unsupported_entities" in r for r in res.reasons)


def test_qc_flags_too_short() -> None:
    out = LabelOutput(summary="Bộ Y tế cho biết.", key_entities=[], confidence=0.9)
    res = run_qc(output=out, source_text=SOURCE, cfg=_cfg(min_words=20))
    assert not res.passed
    assert any("too_short" in r for r in res.reasons)


def test_qc_flags_refusal() -> None:
    out = LabelOutput(
        summary="Không thể tóm tắt.",
        key_entities=[],
        confidence=0.0,
        refusal_reason="content_too_short",
    )
    res = run_qc(output=out, source_text=SOURCE, cfg=_cfg(min_words=1))
    assert not res.passed
    assert any("llm_refusal" in r for r in res.reasons)


def test_qc_handles_fuzzy_entity_match() -> None:
    # Slightly different casing/diacritics still count as a match.
    out = LabelOutput(
        summary=(
            "Bộ Y tế cho biết TP HCM ghi nhận 1.234 ca Covid-19 mới ngày 12/04. "
            "Chủ tịch Nguyễn Văn  A chỉ đạo 5 biện pháp khẩn."
        ),
        key_entities=[],
        confidence=0.9,
    )
    res = run_qc(output=out, source_text=SOURCE, cfg=_cfg())
    assert res.passed, res.reasons
