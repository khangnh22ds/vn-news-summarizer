"""Unit tests for timezone helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

from vn_news_common.time_utils import assume_utc, to_utc, utcnow


def test_utcnow_is_aware() -> None:
    now = utcnow()
    assert now.tzinfo is not None


def test_to_utc_passes_through_none() -> None:
    assert to_utc(None) is None


def test_to_utc_attaches_utc_to_naive_datetime() -> None:
    naive = datetime(2025, 1, 1, 12, 0, 0)
    out = to_utc(naive)
    assert out is not None
    assert out.tzinfo == UTC


def test_to_utc_converts_aware_datetime() -> None:
    aware = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone(timedelta(hours=7)))
    out = to_utc(aware)
    assert out is not None
    assert out.tzinfo == UTC
    # 12:00 +07:00 → 05:00 UTC
    assert out.hour == 5


def test_to_utc_parses_iso_string() -> None:
    out = to_utc("2025-01-01T12:00:00Z")
    assert out is not None
    assert out.year == 2025


def test_to_utc_handles_unparseable_string() -> None:
    assert to_utc("not a date") is None


def test_assume_utc_keeps_aware() -> None:
    aware = datetime(2025, 1, 1, tzinfo=UTC)
    assert assume_utc(aware) == aware
