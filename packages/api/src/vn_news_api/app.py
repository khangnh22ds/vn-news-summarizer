"""FastAPI application entrypoint.

Routes:

- ``GET /``, ``GET /healthz`` — liveness / metadata.
- ``POST /summarize`` — abstractive summary for raw text or a news URL,
  backed by ``ViT5Summarizer`` loaded from ``settings.model_path`` (an
  HF Hub adapter repo by default).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vn_news_common.settings import get_settings

from vn_news_api.routers.summarize import router as summarize_router

app = FastAPI(
    title="vn-news-summarizer",
    version="0.1.0",
    description="Vietnamese real-time news summarization (research/educational).",
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.api_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    version: str
    model_id: str


@app.get("/healthz", response_model=HealthResponse, tags=["meta"])
async def healthz() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(
        status="ok",
        version=app.version,
        model_id=_settings.model_path,
    )


@app.get("/", tags=["meta"])
async def root() -> dict[str, str]:
    return {
        "name": app.title,
        "version": app.version,
        "docs": "/docs",
        "health": "/healthz",
        "summarize": "POST /summarize",
    }


app.include_router(summarize_router)
