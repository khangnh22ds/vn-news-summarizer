"""Unit tests for URL canonicalization and hashing."""

from __future__ import annotations

from vn_news_common.url_utils import canonicalize_url, url_hash


class TestCanonicalize:
    def test_lowercases_scheme_and_host(self) -> None:
        assert canonicalize_url("HTTPS://VnExpress.NET/foo") == "https://vnexpress.net/foo"

    def test_drops_fragment(self) -> None:
        url = "https://vnexpress.net/article-123.html#top"
        assert "#" not in canonicalize_url(url)

    def test_strips_utm_params(self) -> None:
        url = "https://vnexpress.net/foo?utm_source=fb&utm_medium=social&id=42"
        canon = canonicalize_url(url)
        assert "utm_source" not in canon
        assert "utm_medium" not in canon
        assert "id=42" in canon

    def test_strips_fbclid_gclid(self) -> None:
        url = "https://vnexpress.net/foo?fbclid=abc&gclid=xyz&q=news"
        canon = canonicalize_url(url)
        assert "fbclid" not in canon
        assert "gclid" not in canon
        assert "q=news" in canon

    def test_sorts_query_params_for_idempotency(self) -> None:
        a = canonicalize_url("https://x.com/p?b=2&a=1")
        b = canonicalize_url("https://x.com/p?a=1&b=2")
        assert a == b

    def test_idempotent(self) -> None:
        url = "https://VnExpress.net/Foo/Bar?utm_source=x&id=1#frag"
        once = canonicalize_url(url)
        twice = canonicalize_url(once)
        assert once == twice


class TestUrlHash:
    def test_equal_urls_have_equal_hash(self) -> None:
        a = url_hash("https://vnexpress.net/a.html")
        b = url_hash("HTTPS://VnExpress.net/a.html")
        assert a == b

    def test_different_urls_have_different_hash(self) -> None:
        a = url_hash("https://vnexpress.net/a.html")
        b = url_hash("https://vnexpress.net/b.html")
        assert a != b

    def test_hex_format_and_length(self) -> None:
        h = url_hash("https://x.com/")
        assert len(h) == 32
        int(h, 16)  # must be valid hex
