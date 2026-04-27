"""Deduplication helpers.

Layered strategy (called from :mod:`pipeline`):

* **L1 (URL)**: by canonical ``url_hash`` — handled at SQL ``UNIQUE``
  constraint level.
* **L2 (content)**: 64-bit SimHash with Hamming distance ≤ 3.

The L2 check is implemented in Python so we can support both SQLite
and Postgres. For larger deployments this can be moved into a dedicated
index table.
"""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from vn_news_common.models import Article
from vn_news_common.text import is_near_duplicate, simhash64
from vn_news_common.time_utils import utcnow


async def find_near_duplicate(
    session: AsyncSession,
    *,
    text: str,
    within: timedelta = timedelta(days=2),
    threshold: int = 3,
) -> int | None:
    """Return the ID of an existing near-duplicate article, if any.

    We restrict the candidate window to recently fetched rows to keep
    the comparison set small.
    """
    if not text:
        return None
    needle = simhash64(text)
    cutoff = utcnow() - within
    stmt = (
        select(Article.id, Article.simhash)
        .where(Article.fetched_at >= cutoff)
        .where(Article.simhash.is_not(None))
    )
    result = await session.execute(stmt)
    for row_id, h in result.all():
        if h is None:
            continue
        if is_near_duplicate(needle, int(h), threshold=threshold):
            return int(row_id)
    return None
