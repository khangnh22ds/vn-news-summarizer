"""Datetime helpers — UTC-only inside the system."""

from __future__ import annotations

from datetime import UTC, datetime

from dateutil import parser as dateutil_parser


def utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def to_utc(value: datetime | str | None) -> datetime | None:
    """Normalise an arbitrary datetime/string input to timezone-aware UTC.

    - ``None`` is passed through.
    - Naive datetimes are assumed to be in UTC (publishers often omit tz);
      this is consistent with how ``feedparser`` reports times.
    - Strings are parsed via ``dateutil`` (handles RFC 822, ISO 8601, etc.).
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed: datetime = dateutil_parser.parse(value)
        except (ValueError, TypeError, OverflowError):
            return None
    else:
        parsed = value
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def assume_utc(value: datetime) -> datetime:
    """Attach UTC tz to a naive datetime; pass through aware ones unchanged."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
