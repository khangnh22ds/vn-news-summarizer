"""Smoke tests verifying the package skeleton imports correctly.

These exist so `make test` is non-trivially green from TICKET-001 onward.
Real test suites are added per package in later tickets.
"""

from __future__ import annotations

import vn_news_api
import vn_news_common
import vn_news_crawler
import vn_news_inference
import vn_news_labeling
import vn_news_training


def test_packages_have_versions() -> None:
    for pkg in (
        vn_news_common,
        vn_news_crawler,
        vn_news_labeling,
        vn_news_training,
        vn_news_inference,
        vn_news_api,
    ):
        assert isinstance(pkg.__version__, str)
        assert pkg.__version__ != ""
