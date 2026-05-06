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
        "entity_fuzzy_min_ratio": 0.85,
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


# ---------------------------------------------------------------- relax cases
#
# The smoke-test on 50 articles surfaced 11 false-positive QC failures the
# strict heuristics could not distinguish from real factuality issues. The
# tests below pin down the relaxed behaviour: each one exercises a concrete
# failure mode the strict pre-1.2.0 QC was rejecting incorrectly, and is
# paired with a "still-fails" counterexample so we don't trade safety for
# noise reduction.


def test_qc_passes_punctuation_collapsed_score() -> None:
    """``2-0`` should match ``2 0`` written across whitespace in the source."""
    source = (
        "Nottingham Forest đã ghi 2 bàn ngay trong hiệp một, vươn lên dẫn 2 0 "
        "trước Chelsea trước khi đội khách rút ngắn cách biệt."
    )
    out = LabelOutput(
        summary=(
            "Nottingham Forest dẫn trước Chelsea 2-0 sau hiệp một nhờ hai bàn "
            "thắng liên tiếp, trước khi Chelsea rút ngắn cách biệt ở hiệp hai."
        ),
        key_entities=["Chelsea", "Nottingham Forest"],
        confidence=0.9,
    )
    res = run_qc(output=out, source_text=source, cfg=_cfg(max_words=40))
    assert res.passed, res.reasons


def test_qc_passes_year_range_via_digit_groups() -> None:
    """``2019-2024`` should match a source listing each year separately."""
    source = (
        "Hiếu đã thi tốt nghiệp THPT lần đầu năm 2019, các năm tiếp theo 2020, "
        "2021, 2022, 2023 và 2024, lần nào cũng đạt điểm cao môn toán."
    )
    out = LabelOutput(
        summary=(
            "Đặng Minh Hiếu đã sáu lần thi tốt nghiệp THPT trong giai đoạn "
            "2019-2024 và năm nay tham gia kỳ thi lần thứ bảy."
        ),
        key_entities=["Đặng Minh Hiếu"],
        confidence=0.9,
    )
    res = run_qc(output=out, source_text=source, cfg=_cfg(max_words=40))
    assert res.passed, res.reasons


def test_qc_still_flags_hallucinated_score() -> None:
    """The relax must not whitewash a score that is genuinely wrong."""
    source = (
        "Man City để Everton gỡ hòa 2-3 ở phút bù giờ, đánh mất quyền tự "
        "quyết trong cuộc đua vô địch Premier League."
    )
    out = LabelOutput(
        summary=(
            "Man City hòa Everton 5-5 ở vòng 35 Premier League, đánh mất "
            "quyền tự quyết trong cuộc đua vô địch."
        ),
        key_entities=["Man City", "Everton"],
        confidence=0.9,
    )
    res = run_qc(output=out, source_text=source, cfg=_cfg(max_words=40))
    assert not res.passed
    assert any("unsupported_numbers" in r for r in res.reasons)


def test_qc_passes_entity_via_lastname_fallback() -> None:
    """``HLV Simeone`` should match a body that uses the bare surname."""
    source = (
        "Atletico Madrid của Simeone đã thi đấu kiên cường tại lượt đi và "
        "buộc Arsenal phải chia điểm 1-1 trên sân Metropolitano."
    )
    out = LabelOutput(
        summary=(
            "Arsenal có lợi thế sân nhà Emirates trước Atletico Madrid của "
            "HLV Simeone, sau khi hai đội đã hòa 1-1 ở lượt đi."
        ),
        key_entities=["Atletico Madrid"],
        confidence=0.9,
    )
    res = run_qc(output=out, source_text=source, cfg=_cfg(max_words=40))
    assert res.passed, res.reasons


def test_qc_lastname_fallback_does_not_carry_short_helpers() -> None:
    """``HLV Khac`` must still fail when neither token is a real surname.

    Without the >=4 character guard, the helper word ``HLV`` would be a
    free pass against any source containing it, so we explicitly assert
    that an unfamiliar surname combined with a known helper still fails.
    """
    source = "HLV Pep Guardiola dẫn dắt Man City qua một trận đấu khó khăn."
    out = LabelOutput(
        summary=(
            "Theo HLV Khac, Man City đã chơi một trận đấu khó khăn và phải "
            "chia điểm với đội chủ nhà sau hiệp hai bùng nổ."
        ),
        key_entities=["Man City"],
        confidence=0.9,
    )
    res = run_qc(output=out, source_text=source, cfg=_cfg(max_words=40))
    assert not res.passed
    assert any("unsupported_entities" in r for r in res.reasons)
