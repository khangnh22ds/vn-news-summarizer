"""Polite async HTTP client with per-host rate limiting + retries.

Behaviour:

* Honours a global "requests per second per host" cap (default 1.0).
* Adds a small random jitter so concurrent crawls don't synchronise.
* Retries 5xx + 429 with exponential backoff via :mod:`tenacity`.
* Always sends our project User-Agent.
"""

from __future__ import annotations

import asyncio
import random
import time
import urllib.parse
from collections.abc import Mapping

import httpx
from loguru import logger
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class _HostLimiter:
    """Token-style limiter giving min interval between requests per host."""

    def __init__(self, rps: float) -> None:
        self._min_interval = 1.0 / rps if rps > 0 else 0.0
        self._last: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def acquire(self, host: str) -> None:
        if self._min_interval <= 0:
            return
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            now = time.monotonic()
            last = self._last.get(host, 0.0)
            wait = self._min_interval - (now - last)
            if wait > 0:
                await asyncio.sleep(wait + random.uniform(0, 0.05))
            self._last[host] = time.monotonic()


class PoliteClient:
    """Wraps :class:`httpx.AsyncClient` with rate limiting + retries."""

    def __init__(
        self,
        *,
        user_agent: str,
        timeout_s: float = 20.0,
        rps_per_host: float = 1.0,
        max_retries: int = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        headers = {"User-Agent": user_agent, "Accept-Language": "vi,en;q=0.7"}
        self._client = client or httpx.AsyncClient(
            timeout=timeout_s,
            headers=headers,
            follow_redirects=True,
        )
        self._limiter = _HostLimiter(rps_per_host)
        self._max_retries = max_retries

    async def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        host = urllib.parse.urlsplit(url).netloc.lower()
        await self._limiter.acquire(host)
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_retries + 1),
            wait=wait_exponential(multiplier=1, min=1, max=15),
            retry=retry_if_exception_type((httpx.HTTPError,)),
            reraise=True,
        ):
            with attempt:
                res = await self._client.get(url, headers=dict(headers) if headers else None)
                if res.status_code == 429 or 500 <= res.status_code < 600:
                    logger.warning("Retryable {} for {}", res.status_code, url)
                    raise httpx.HTTPStatusError(
                        f"server status {res.status_code}",
                        request=res.request,
                        response=res,
                    )
                return res
        raise RuntimeError("unreachable")  # pragma: no cover

    async def aclose(self) -> None:
        await self._client.aclose()
