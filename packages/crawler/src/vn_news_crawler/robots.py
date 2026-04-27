"""Cached, async-friendly robots.txt checker.

We only consult the registrable host (e.g. ``vnexpress.net``) — not the
URL path's host — so subdomains are checked individually.

The cache is process-local and TTL-based (default 1 hour). If fetching
``robots.txt`` fails (network error, 5xx) we conservatively allow access
but log a warning; for clear 4xx (404 = no robots) we allow.
"""

from __future__ import annotations

import time
import urllib.parse
from dataclasses import dataclass
from urllib.robotparser import RobotFileParser

import httpx
from loguru import logger

DEFAULT_TTL_S = 3600.0


@dataclass
class _Entry:
    parser: RobotFileParser
    fetched_at: float
    fetch_ok: bool


class RobotsCache:
    """Per-host cache for robots.txt rules."""

    def __init__(
        self,
        *,
        user_agent: str,
        ttl_s: float = DEFAULT_TTL_S,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._user_agent = user_agent
        self._ttl = ttl_s
        self._cache: dict[str, _Entry] = {}
        self._client = client

    async def _client_or_default(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=10.0,
                headers={"User-Agent": self._user_agent},
                follow_redirects=True,
            )
        return self._client

    async def _fetch(self, host: str) -> _Entry:
        url = f"https://{host}/robots.txt"
        parser = RobotFileParser()
        parser.set_url(url)
        try:
            client = await self._client_or_default()
            res = await client.get(url)
            if res.status_code == 200:
                parser.parse(res.text.splitlines())
                ok = True
            elif 400 <= res.status_code < 500:
                # No robots.txt → assume open.
                parser.parse(["User-agent: *", "Allow: /"])
                ok = True
            else:
                logger.warning(
                    "robots.txt for {} returned {} — conservatively allowing", host, res.status_code
                )
                parser.parse(["User-agent: *", "Allow: /"])
                ok = False
        except (httpx.HTTPError, OSError) as exc:
            logger.warning("robots.txt fetch failed for {}: {}", host, exc)
            parser.parse(["User-agent: *", "Allow: /"])
            ok = False
        return _Entry(parser=parser, fetched_at=time.monotonic(), fetch_ok=ok)

    async def can_fetch(self, url: str) -> bool:
        """Return True if our user-agent may fetch ``url``."""
        host = urllib.parse.urlsplit(url).netloc.lower()
        if not host:
            return False
        entry = self._cache.get(host)
        if entry is None or (time.monotonic() - entry.fetched_at) > self._ttl:
            entry = await self._fetch(host)
            self._cache[host] = entry
        return entry.parser.can_fetch(self._user_agent, url)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
