"""Shared pytest fixtures."""

from __future__ import annotations

import os

# Force the regex fallback in :mod:`vn_news_training.baselines.tokenizer`
# during the test suite. ``underthesea`` ships C extensions that segfault
# on interpreter shutdown when imported and torn down repeatedly, which
# would otherwise turn pytest's exit code into 139.
os.environ.setdefault("VN_NEWS_USE_UNDERTHESEA", "0")

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from vn_news_common.db import dispose_engine, get_engine, get_session_maker
from vn_news_common.models import Base
from vn_news_common.settings import reset_settings


@pytest_asyncio.fixture
async def db_session(tmp_path: Path) -> AsyncIterator[AsyncSession]:
    """Provision a fresh on-disk SQLite database for each test."""
    db_path = tmp_path / "test.db"
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path}"
    reset_settings()
    await dispose_engine()

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    maker = get_session_maker()
    async with maker() as session:
        yield session

    await dispose_engine()
    os.environ.pop("DATABASE_URL", None)
    reset_settings()


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"
