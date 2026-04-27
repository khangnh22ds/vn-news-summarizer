"""End-to-end test for the eval CLI: write a tiny JSONL → run eval → check
metrics dict and exit code."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

import run_eval  # type: ignore[import-not-found]

SOURCE_TEXT = (
    "Hôm 12/04, Bộ Y tế cho biết TP HCM ghi nhận 1.234 ca mắc Covid-19 mới. "
    "Chủ tịch Nguyễn Văn A chỉ đạo triển khai 5 biện pháp cấp bách để phòng "
    "dịch tại các quận trung tâm. Các bệnh viện tuyến cuối được yêu cầu sẵn "
    "sàng giường bệnh và oxy y tế."
)
SUMMARY_TEXT = (
    "Bộ Y tế cho biết TP HCM ghi nhận 1.234 ca Covid-19 mới ngày 12/04. "
    "Chủ tịch Nguyễn Văn A triển khai 5 biện pháp cấp bách."
)


def _make_dataset(root: Path, name: str = "v1") -> Path:
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "article_id": i,
            "source": "vnexpress",
            "url": f"https://vnexpress.net/{i}.html",
            "title": f"Tin số {i}",
            "category": "thoi-su",
            "published_at": None,
            "content_text": SOURCE_TEXT,
            "summary": SUMMARY_TEXT,
            "prompt_version": "1.0.0",
        }
        for i in range(3)
    ]
    for split in ("train", "val", "test"):
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


@pytest.mark.parametrize("baseline", ["lexrank", "textrank"])
def test_run_eval_cli_prints_metrics(tmp_path: Path, baseline: str) -> None:
    _make_dataset(tmp_path, name="v1")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = run_eval.main(
            [
                "--baseline",
                baseline,
                "--dataset",
                "v1",
                "--dataset-root",
                str(tmp_path),
                "--split",
                "test",
                "--no-mlflow",
            ]
        )
    assert rc == 0
    metrics = json.loads(buf.getvalue())
    assert "rouge1_f1" in metrics
    assert "rougeL_f1" in metrics
    assert metrics["n"] == 3
    # Sanity: against an article that contains the reference verbatim,
    # ROUGE-1 should be well above zero.
    assert metrics["rouge1_f1"] > 0.0


def test_run_eval_cli_missing_dataset_returns_2(tmp_path: Path) -> None:
    rc = run_eval.main(
        [
            "--baseline",
            "lexrank",
            "--dataset",
            "does-not-exist",
            "--dataset-root",
            str(tmp_path),
            "--no-mlflow",
        ]
    )
    assert rc == 2


def test_run_eval_cli_with_limit_truncates(tmp_path: Path) -> None:
    _make_dataset(tmp_path, name="v1")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = run_eval.main(
            [
                "--baseline",
                "lexrank",
                "--dataset",
                "v1",
                "--dataset-root",
                str(tmp_path),
                "--split",
                "test",
                "--limit",
                "1",
                "--no-mlflow",
            ]
        )
    assert rc == 0
    metrics = json.loads(buf.getvalue())
    assert metrics["n"] == 1
