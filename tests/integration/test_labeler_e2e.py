"""Integration tests for the labeler pipeline against a fresh SQLite DB."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from vn_news_common.enums import ArticleStatus
from vn_news_common.models import Article, DatasetVersion, Label, Source
from vn_news_labeling import (
    GenerationParams,
    Prompt,
    VertexLabeler,
    build_dataset_version,
    label_batch,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = REPO_ROOT / "configs" / "prompts" / "summarize_v1.yaml"


SOURCE_TEXT = (
    "Hôm 12/04, Bộ Y tế cho biết TP HCM ghi nhận 1.234 ca mắc Covid-19 mới. "
    "Chủ tịch Nguyễn Văn A chỉ đạo triển khai 5 biện pháp cấp bách "
    "để phòng dịch tại các quận trung tâm. Các bệnh viện tuyến cuối được yêu "
    "cầu sẵn sàng giường bệnh và oxy y tế. Sở Y tế thành phố sẽ giám sát "
    "việc tuân thủ tại các khu công nghiệp và trường học."
)

GOOD_LABEL = (
    '{"summary": "Bộ Y tế cho biết TP HCM ghi nhận 1.234 ca Covid-19 '
    "mới ngày 12/04. Chủ tịch Nguyễn Văn A chỉ đạo triển khai 5 biện "
    'pháp cấp bách để phòng dịch.", '
    '"key_entities": ["TP HCM", "Bộ Y tế", "Nguyễn Văn A"], '
    '"confidence": 0.92, "refusal_reason": null}'
)

BAD_LABEL = (
    '{"summary": "Hà Nội ghi nhận 9999 ca Covid-19 mới ngày 12/04. '
    'Chủ tịch Trần Văn B chỉ đạo các biện pháp cấp bách.", '
    '"key_entities": ["Hà Nội"], "confidence": 0.5, "refusal_reason": null}'
)


async def _seed(session: AsyncSession, *, n: int = 3) -> list[int]:
    src = Source(
        source_id="vnexpress",
        name="VnExpress",
        domain="vnexpress.net",
        rss_urls=["https://vnexpress.net/rss/tin-moi-nhat.rss"],
    )
    session.add(src)
    await session.flush()
    article_ids: list[int] = []
    for i in range(n):
        a = Article(
            source_fk=src.id,
            url=f"https://vnexpress.net/news/{i}.html",
            url_hash=f"h{i:032d}",
            title=f"Tiêu đề {i}",
            category="thoi-su",
            content_text=SOURCE_TEXT,
            word_count=80,
            status=ArticleStatus.CLEANED,
        )
        a.source = src
        session.add(a)
        await session.flush()
        article_ids.append(a.id)
    await session.commit()
    return article_ids


@pytest.mark.asyncio
async def test_label_batch_persists_good_and_bad_labels(db_session: AsyncSession) -> None:
    await _seed(db_session, n=2)

    prompt = Prompt.from_yaml(PROMPT_PATH)
    # Loosen QC bounds so the canned summary fits.
    prompt.qc.min_words = 10
    prompt.qc.max_words = 80
    prompt.qc.min_sentences = 1
    prompt.qc.max_sentences = 5

    calls = {"n": 0}

    def alternating(_system: str, _user: str) -> str:
        calls["n"] += 1
        return GOOD_LABEL if calls["n"] == 1 else BAD_LABEL

    labeler = VertexLabeler(
        project=None,
        location="asia-southeast1",
        model_name="gemini-2.0-flash-001",
        params=GenerationParams(),
        override_callable=alternating,
    )

    stats = await label_batch(prompt=prompt, labeler=labeler, limit=10)
    assert stats.requested == 2
    assert stats.labeled == 2
    assert stats.qc_passed == 1
    assert stats.qc_failed == 1
    assert stats.llm_errors == 0

    # DB now has two Label rows; one passed, one failed.
    labels = (await db_session.execute(select(Label))).scalars().all()
    assert len(labels) == 2
    assert {bool(label.qc_passed) for label in labels} == {True, False}

    # The article whose label passed should be SUMMARIZED.
    db_session.expire_all()
    arts = (await db_session.execute(select(Article))).scalars().all()
    statuses = {a.status for a in arts}
    assert ArticleStatus.SUMMARIZED in statuses
    assert ArticleStatus.CLEANED in statuses


@pytest.mark.asyncio
async def test_label_batch_skips_already_labeled(db_session: AsyncSession) -> None:
    await _seed(db_session, n=1)

    prompt = Prompt.from_yaml(PROMPT_PATH)
    prompt.qc.min_words = 10
    prompt.qc.max_words = 80

    labeler = VertexLabeler(
        project=None,
        location="asia-southeast1",
        model_name="gemini-2.0-flash-001",
        params=GenerationParams(),
        override_callable=lambda *_: GOOD_LABEL,
    )

    s1 = await label_batch(prompt=prompt, labeler=labeler, limit=10)
    assert s1.labeled == 1

    # Second pass should pick zero (already labeled at this prompt version).
    s2 = await label_batch(prompt=prompt, labeler=labeler, limit=10)
    assert s2.requested == 0
    assert s2.labeled == 0


@pytest.mark.asyncio
async def test_export_dataset_version(db_session: AsyncSession, tmp_path: Path) -> None:
    await _seed(db_session, n=4)

    prompt = Prompt.from_yaml(PROMPT_PATH)
    prompt.qc.min_words = 10
    prompt.qc.max_words = 80

    labeler = VertexLabeler(
        project=None,
        location="asia-southeast1",
        model_name="gemini-2.0-flash-001",
        params=GenerationParams(),
        override_callable=lambda *_: GOOD_LABEL,
    )
    await label_batch(prompt=prompt, labeler=labeler, limit=10)

    out_root = tmp_path / "datasets"
    ds = await build_dataset_version(
        name="vtest",
        prompt_version=prompt.version,
        out_root=out_root,
    )
    assert ds.total == 4
    # Three split files exist with correct row counts.
    for split in ("train", "val", "test"):
        path = out_root / "vtest" / f"{split}.jsonl"
        assert path.exists()
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        assert len(rows) == getattr(ds, split)
        for r in rows:
            assert r["summary"]
            assert r["content_text"]
            assert r["prompt_version"] == prompt.version

    # DatasetVersion row was inserted.
    versions = (await db_session.execute(select(DatasetVersion))).scalars().all()
    assert len(versions) == 1
    assert versions[0].name == "vtest"
    assert len(versions[0].train_ids) + len(versions[0].val_ids) + len(versions[0].test_ids) == 4
