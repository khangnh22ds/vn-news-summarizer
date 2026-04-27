"""Minimal FastAPI app placeholder for TICKET-001.

Real routers (articles, sources, admin) will be added in Phase 5.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="vn-news-summarizer",
    version="0.1.0",
    description="Vietnamese real-time news summarization (research/educational).",
)


class HealthResponse(BaseModel):
    status: str
    version: str


@app.get("/healthz", response_model=HealthResponse, tags=["meta"])
async def healthz() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(status="ok", version=app.version)


@app.get("/", tags=["meta"])
async def root() -> dict[str, str]:
    return {
        "name": app.title,
        "version": app.version,
        "docs": "/docs",
        "health": "/healthz",
    }
