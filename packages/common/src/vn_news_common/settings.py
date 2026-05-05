"""Pydantic-settings configuration shared across all packages.

Reads from environment variables (and ``.env`` for local dev). Each field
maps directly onto a key documented in ``.env.example``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    """Application-wide settings."""

    # ---- Database -------------------------------------------------------
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/app.db",
        description="SQLAlchemy async DSN.",
    )

    # ---- Redis ----------------------------------------------------------
    redis_url: str = Field(default="redis://localhost:6379/0")

    # ---- Vertex AI ------------------------------------------------------
    google_cloud_project: str | None = Field(default=None)
    google_cloud_location: str = Field(default="asia-southeast1")
    google_application_credentials: str | None = Field(default=None)
    vertex_llm_model: str = Field(default="gemini-2.5-flash")

    # ---- Crawler --------------------------------------------------------
    crawler_user_agent: str = Field(
        default=(
            "vn-news-summarizer-research/0.1 (+https://github.com/khangnh22ds/vn-news-summarizer)"
        ),
    )
    crawler_requests_per_second: float = Field(default=1.0, ge=0.1, le=10.0)
    crawler_timeout_seconds: float = Field(default=20.0, ge=1.0, le=120.0)
    crawler_max_retries: int = Field(default=3, ge=0, le=10)

    # ---- API ------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_log_level: Literal["debug", "info", "warning", "error"] = "info"
    admin_token: str = Field(default="change-me")

    # ---- Inference ------------------------------------------------------
    model_path: str = Field(default="./models/vit5-news-v1")
    enable_llm_fallback: bool = Field(default=True)

    # ---- MLflow ---------------------------------------------------------
    mlflow_tracking_uri: str = Field(default="file:./mlruns")

    # ---- Misc -----------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    tz: str = Field(default="Asia/Ho_Chi_Minh")

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached Settings instance.

    Tests can override by clearing the cache via :func:`reset_settings`.
    """
    global _settings  # noqa: PLW0603 — module-level cache by design
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset the cached settings instance (intended for tests)."""
    global _settings  # noqa: PLW0603
    _settings = None
