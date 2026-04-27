"""Unit tests for the RobotsCache."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest
from vn_news_crawler.robots import RobotsCache

_Handler = Callable[[httpx.Request], httpx.Response]


def _mock_client(handler: _Handler) -> httpx.AsyncClient:
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(
        transport=transport,
        headers={"User-Agent": "test-agent"},
    )


@pytest.mark.asyncio
async def test_allows_when_robots_allows() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="User-agent: *\nAllow: /\n")

    cache = RobotsCache(user_agent="test-agent", client=_mock_client(handler))
    assert await cache.can_fetch("https://example.com/article") is True
    await cache.aclose()


@pytest.mark.asyncio
async def test_disallows_specific_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")

    cache = RobotsCache(user_agent="test-agent", client=_mock_client(handler))
    assert await cache.can_fetch("https://example.com/private/secret") is False
    assert await cache.can_fetch("https://example.com/public") is True
    await cache.aclose()


@pytest.mark.asyncio
async def test_404_treated_as_allow() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    cache = RobotsCache(user_agent="test-agent", client=_mock_client(handler))
    assert await cache.can_fetch("https://example.com/anything") is True
    await cache.aclose()


@pytest.mark.asyncio
async def test_network_error_allows_conservatively() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    cache = RobotsCache(user_agent="test-agent", client=_mock_client(handler))
    assert await cache.can_fetch("https://example.com/foo") is True
    await cache.aclose()


@pytest.mark.asyncio
async def test_cache_is_used_across_calls() -> None:
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        return httpx.Response(200, text="User-agent: *\nAllow: /\n")

    cache = RobotsCache(user_agent="test-agent", client=_mock_client(handler))
    await cache.can_fetch("https://example.com/a")
    await cache.can_fetch("https://example.com/b")
    await cache.can_fetch("https://example.com/c")
    assert counter["n"] == 1, "robots.txt should be fetched only once per host within TTL"
    await cache.aclose()
