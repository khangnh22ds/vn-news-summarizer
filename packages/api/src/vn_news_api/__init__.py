"""FastAPI application package.

Real router implementations land in TICKET-007 (Phase 5). For TICKET-001
we ship a minimal placeholder app so `make api` boots and `/healthz`
responds — useful as a smoke test for the dev environment.
"""

from __future__ import annotations

from vn_news_api.app import app

__version__ = "0.1.0"
__all__ = ["__version__", "app"]
