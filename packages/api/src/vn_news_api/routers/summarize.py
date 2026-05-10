"""``POST /summarize`` — abstractive summary for arbitrary Vietnamese text or a news URL.

Two input modes are accepted on the same endpoint:

- ``{"text": "..."}``: summarise the raw text directly.
- ``{"url": "https://..."}``: fetch the URL, extract the article body
  via the same trafilatura-first pipeline that powers the offline
  crawler, then summarise that body.

Exactly one of ``text`` / ``url`` must be provided.
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field, HttpUrl, model_validator
from starlette.concurrency import run_in_threadpool
from vn_news_common.settings import get_settings

from vn_news_api.services.article_fetcher import (
    ArticleFetchError,
    fetch_and_extract,
)
from vn_news_api.services.summarizer_singleton import get_summarizer

router = APIRouter(tags=["summarize"])


class SummarizeRequest(BaseModel):
    text: Annotated[str | None, Field(default=None, min_length=1)]
    url: HttpUrl | None = None

    @model_validator(mode="after")
    def _exactly_one_input(self) -> SummarizeRequest:
        has_text = self.text is not None and self.text.strip() != ""
        has_url = self.url is not None
        if has_text == has_url:  # both or neither
            msg = "exactly one of `text` or `url` must be provided"
            raise ValueError(msg)
        return self


class SummarizeResponse(BaseModel):
    summary: str
    model_id: str
    source_url: str | None = None
    source_title: str | None = None
    input_chars: int
    summary_chars: int
    elapsed_ms: int


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    status_code=status.HTTP_200_OK,
)
async def summarize(req: SummarizeRequest) -> SummarizeResponse:
    settings = get_settings()
    started = time.perf_counter()

    source_url: str | None = None
    source_title: str | None = None
    if req.url is not None:
        source_url = str(req.url)
        try:
            article = await run_in_threadpool(fetch_and_extract, source_url)
        except ArticleFetchError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={"reason": exc.reason, "message": str(exc)},
            ) from exc
        text = article.content_text
        source_title = article.title
    else:
        # Validator above guarantees text is non-empty when url is None.
        assert req.text is not None
        text = req.text.strip()

    if len(text) > settings.summarize_max_input_chars:
        # Keep the head; tail truncation is fine for summarisation.
        logger.info(
            "summarize input truncated from {} to {} chars",
            len(text),
            settings.summarize_max_input_chars,
        )
        text = text[: settings.summarize_max_input_chars]

    summarizer = get_summarizer()
    summary = await run_in_threadpool(summarizer.summarize, text)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return SummarizeResponse(
        summary=summary,
        model_id=settings.model_path,
        source_url=source_url,
        source_title=source_title,
        input_chars=len(text),
        summary_chars=len(summary),
        elapsed_ms=elapsed_ms,
    )
