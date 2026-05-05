"""Unit tests for text utilities and SimHash."""

from __future__ import annotations

from vn_news_common.text import (
    hamming,
    is_near_duplicate,
    normalize_text,
    simhash64,
    word_count,
)


class TestNormalize:
    def test_collapses_whitespace(self) -> None:
        assert normalize_text("a   b\n\n\nc") == "a b c"

    def test_strips(self) -> None:
        assert normalize_text("   hello world   ") == "hello world"

    def test_handles_empty(self) -> None:
        assert normalize_text("") == ""


class TestWordCount:
    def test_basic(self) -> None:
        assert word_count("một hai ba bốn năm") == 5

    def test_empty(self) -> None:
        assert word_count("") == 0


class TestSimhash:
    def test_identical_texts_have_zero_distance(self) -> None:
        text = "Việt Nam giành huy chương vàng tại SEA Games 2025."
        assert hamming(simhash64(text), simhash64(text)) == 0

    def test_minor_changes_have_smaller_distance_than_unrelated(self) -> None:
        a = "Việt Nam giành huy chương vàng tại SEA Games 2025 với thành tích xuất sắc."
        b = "Việt Nam đã giành huy chương vàng tại SEA Games 2025 thành tích xuất sắc."
        c = "Tổng thống Mỹ ký sắc lệnh hành pháp mới về thương mại quốc tế."
        d_similar = hamming(simhash64(a), simhash64(b))
        d_unrelated = hamming(simhash64(a), simhash64(c))
        assert d_similar < d_unrelated

    def test_unrelated_texts_have_large_distance(self) -> None:
        a = "Bóng đá Việt Nam chuẩn bị cho vòng loại World Cup."
        b = "Tổng thống Mỹ ký sắc lệnh hành pháp mới về thương mại."
        assert hamming(simhash64(a), simhash64(b)) > 10

    def test_is_near_duplicate(self) -> None:
        text = "Tin tức về kinh tế Việt Nam quý 2 năm 2025 với nhiều biến động."
        assert is_near_duplicate(simhash64(text), simhash64(text), threshold=3)

    def test_handles_heavily_repeated_ngram(self) -> None:
        """Pages with a single n-gram repeated >50 times previously triggered
        ``OverflowError: Python integer N out of bounds for uint8`` from the
        upstream ``simhash`` library on numpy >= 1.24. simhash64 must stay
        overflow-free and return a stable 64-bit integer.
        """
        # ~500 repetitions of the same token -> per-feature weight far above
        # the upstream cutoff of 50.
        text = "Phường Bến Thành Trường THPT " * 500
        value = simhash64(text)
        assert isinstance(value, int)
        # Deterministic: calling twice returns the same fingerprint.
        assert simhash64(text) == value
        # And fits in signed 64-bit.
        assert -(1 << 63) <= value < (1 << 63)
