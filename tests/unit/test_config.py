"""Unit tests for sources.yaml loader and category mapper."""

from __future__ import annotations

from pathlib import Path

from vn_news_crawler.config import find_canonical_category, load_sources_config

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_load_real_sources_yaml() -> None:
    cfg = load_sources_config(REPO_ROOT / "configs" / "sources.yaml")
    # 8 sources configured per project plan.
    assert len(cfg.sources) == 8
    enabled = cfg.enabled()
    assert len(enabled) >= 3
    ids = {s.id for s in cfg.sources}
    expected = {
        "vnexpress",
        "tuoitre",
        "thanhnien",
        "dantri",
        "vietnamnet",
        "znews",
        "vtcnews",
        "laodong",
    }
    assert expected.issubset(ids)


def test_each_source_has_at_least_one_rss_feed() -> None:
    cfg = load_sources_config(REPO_ROOT / "configs" / "sources.yaml")
    for src in cfg.sources:
        assert len(src.rss) >= 1, f"{src.id} has no RSS feeds"


def test_find_canonical_category_substring_match() -> None:
    canonical = {
        "thoi_su": ["thoi-su", "thời sự"],
        "kinh_doanh": ["kinh-doanh", "kinh tế"],
    }
    assert find_canonical_category("Thời sự", canonical=canonical) == "thoi_su"
    assert find_canonical_category("kinh-doanh", canonical=canonical) == "kinh_doanh"
    assert find_canonical_category("unknown", canonical=canonical) is None
    assert find_canonical_category(None, canonical=canonical) is None


def test_canonical_categories_loaded() -> None:
    cfg = load_sources_config(REPO_ROOT / "configs" / "sources.yaml")
    assert len(cfg.canonical_categories) > 0
