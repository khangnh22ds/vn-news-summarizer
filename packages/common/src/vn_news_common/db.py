"""Async SQLAlchemy engine + session factory.

The engine is lazily constructed so importing :mod:`vn_news_common` does
not require a configured database (matters for tests, scripts, and the
labeling worker which can run without DB).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from vn_news_common.settings import get_settings

_engine: AsyncEngine | None = None
_session_maker: async_sessionmaker[AsyncSession] | None = None


def _make_engine(url: str) -> AsyncEngine:
    kwargs: dict[str, Any] = {"future": True}
    if url.startswith("sqlite"):
        # SQLite-specific: enforce FK constraints and avoid file-locking surprises.
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_size"] = 5
        kwargs["max_overflow"] = 10
        kwargs["pool_pre_ping"] = True
    return create_async_engine(url, **kwargs)


def get_engine() -> AsyncEngine:
    """Return a process-wide singleton :class:`AsyncEngine`."""
    global _engine  # noqa: PLW0603 — module-level cache by design
    if _engine is None:
        _engine = _make_engine(get_settings().database_url)
    return _engine


def get_session_maker() -> async_sessionmaker[AsyncSession]:
    """Return a process-wide :class:`async_sessionmaker`."""
    global _session_maker  # noqa: PLW0603
    if _session_maker is None:
        _session_maker = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_maker


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Provide a transactional scope for a series of operations."""
    sm = get_session_maker()
    async with sm() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def dispose_engine() -> None:
    """Dispose the engine — used in tests + at shutdown."""
    global _engine, _session_maker  # noqa: PLW0603
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_maker = None
