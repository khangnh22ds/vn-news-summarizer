"""Unit tests for the JSONL dataset loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from vn_news_training import Example, load_dataset, load_split


def _write_split(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _row(article_id: int, title: str = "T", summary: str = "S") -> dict[str, object]:
    return {
        "article_id": article_id,
        "source": "vnexpress",
        "url": f"https://x/{article_id}",
        "title": title,
        "category": "thoi-su",
        "published_at": None,
        "content_text": "lorem ipsum",
        "summary": summary,
        "prompt_version": "1.0.0",
    }


def test_load_split_parses_jsonl(tmp_path: Path) -> None:
    _write_split(tmp_path / "test.jsonl", [_row(1), _row(2)])
    rows = load_split(tmp_path, "test")
    assert len(rows) == 2
    assert all(isinstance(r, Example) for r in rows)
    assert rows[0].article_id == 1
    assert rows[1].source == "vnexpress"


def test_load_split_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(_row(1), ensure_ascii=False) + "\n\n\n", encoding="utf-8")
    rows = load_split(tmp_path, "train")
    assert len(rows) == 1


def test_load_split_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_split(tmp_path, "test")


def test_load_dataset_returns_all_three_splits(tmp_path: Path) -> None:
    for s in ("train", "val", "test"):
        _write_split(tmp_path / f"{s}.jsonl", [_row(1)])
    out = load_dataset(tmp_path)
    assert set(out.keys()) == {"train", "val", "test"}
    assert all(len(v) == 1 for v in out.values())


def test_example_handles_missing_optional_fields(tmp_path: Path) -> None:
    raw = {"article_id": 1, "url": "https://x/1"}
    _write_split(tmp_path / "test.jsonl", [raw])
    rows = load_split(tmp_path, "test")
    assert rows[0].article_id == 1
    assert rows[0].title == ""
    assert rows[0].category is None
