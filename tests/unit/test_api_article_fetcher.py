"""Unit tests for ``article_fetcher.fetch_and_extract`` (httpx mocked).

These tests don't require a real network — ``respx`` intercepts the
outbound httpx call. They cover the three error branches plus the
happy path.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from vn_news_api.services.article_fetcher import (
    ArticleFetchError,
    fetch_and_extract,
)

# Long Vietnamese article body so trafilatura's >=50-word threshold is met.
_REAL_BODY = (
    "Hôm nay tại Hà Nội, đội tuyển bóng đá Việt Nam đã có buổi tập kéo dài hơn "
    "hai giờ đồng hồ trên sân Mỹ Đình. Huấn luyện viên trưởng cho biết toàn "
    "đội đang tập trung tối đa cho trận đấu sắp tới với mục tiêu giành ba "
    "điểm trọn vẹn để củng cố vị trí trên bảng xếp hạng. Các cầu thủ trẻ "
    "cũng được trao cơ hội tham gia tập luyện cùng đội tuyển quốc gia. "
    "Buổi tập diễn ra trong bầu không khí khẩn trương nhưng vẫn đậm tinh "
    "thần đoàn kết, sẵn sàng bước vào trận đấu quan trọng cuối tuần này."
)
_HTML_DOC = (
    "<html><head><title>Bài báo mẫu</title></head>"
    f"<body><article><p>{_REAL_BODY}</p></article></body></html>"
)


@respx.mock
def test_fetch_and_extract_happy_path() -> None:
    respx.get("https://example.com/article").mock(return_value=httpx.Response(200, text=_HTML_DOC))
    article = fetch_and_extract("https://example.com/article")
    assert "Hà Nội" in article.content_text
    # Title is best-effort; trafilatura usually picks it up but readability
    # fallback may not, so just assert the field exists with the right type.
    assert article.title is None or isinstance(article.title, str)
    assert article.word_count >= 50


@respx.mock
def test_fetch_and_extract_raises_on_4xx() -> None:
    respx.get("https://example.com/missing").mock(
        return_value=httpx.Response(404, text="not found")
    )
    with pytest.raises(ArticleFetchError) as exc_info:
        fetch_and_extract("https://example.com/missing")
    assert exc_info.value.reason == "upstream_error"


@respx.mock
def test_fetch_and_extract_raises_on_network_error() -> None:
    respx.get("https://example.com/timeout").mock(
        side_effect=httpx.ConnectTimeout("simulated timeout")
    )
    with pytest.raises(ArticleFetchError) as exc_info:
        fetch_and_extract("https://example.com/timeout")
    assert exc_info.value.reason == "fetch_failed"


@respx.mock
def test_fetch_and_extract_raises_on_unparseable_body() -> None:
    """Body shorter than 50 words → both extractors fail → ``extract_failed``."""
    respx.get("https://example.com/sparse").mock(
        return_value=httpx.Response(200, text="<html><body><p>too short</p></body></html>")
    )
    with pytest.raises(ArticleFetchError) as exc_info:
        fetch_and_extract("https://example.com/sparse")
    assert exc_info.value.reason == "extract_failed"
