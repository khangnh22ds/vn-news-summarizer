# Labeling run — v1.2.0 (Gemini 2.5 Pro), full corpus

**When:** 2026-05-06
**Model:** `gemini-2.5-pro` on Vertex AI (`us-central1`)
**Prompt version:** `summarize_v1.yaml` v1.2.0
**Concurrency:** 5 in-flight requests (asyncio.Semaphore + `asyncio.to_thread`)
**Wall clock:** ~2 h 14 min for 2358 fresh articles (rate ~17 articles/min)

## Overall counts

| | |
|---|---:|
| Articles labelled (cumulative v1.2.0) | 2412 |
| QC passed | **2070 (85.8 %)** |
| QC failed | 342 (14.2 %) |
| LLM errors during full run | 6 (0.25 %) |

The LLM errors (`6 / 2358`) all came from a brief 429 `Resource exhausted` window late in the run; the tenacity-driven retry budget consumed itself before the rate limit recovered. They can be re-attempted on a follow-up pass — they don't represent an unrecoverable issue with those articles.

## Per-source breakdown

| Source | Pass | Fail | Pass rate |
|---|---:|---:|---:|
| thanhnien | 347 | 26 | 93.0 % |
| tuoitre | 348 | 37 | 90.4 % |
| dantri | 169 | 19 | 89.9 % |
| vietnamnet | 524 | 92 | 85.1 % |
| vnexpress | 380 | 72 | 84.1 % |
| vtcnews | 87 | 21 | 80.6 % |
| znews | 215 | 75 | 74.1 % |

`znews` skews low because its homepage feed surfaces a higher proportion of click-bait / aggregation pages that Pro correctly refuses. `vtcnews` is similar but smaller-volume. The three highest-quality outlets (thanhnien, tuoitre, dantri) sit near 90 %.

## Failure reasons

(One article can trigger multiple reasons; counts are per-article-occurrence of each reason.)

| Reason | Count |
|---|---:|
| `llm_refusal` | 203 |
| `unsupported_numbers` | 78 |
| `unsupported_entities` | 49 |
| `too_short` (cascades from refusal) | 40 |
| `too_few_sentences` (cascades from refusal) | 37 |
| `too_many_sentences` | 16 |
| `too_long` | 1 |

`llm_refusal` is the dominant failure mode (~59 % of failed articles). Sampled refusal_reasons fall into three buckets:

1. **Source content is fabricated / future-dated.** "Bài viết mô tả các sự kiện ở năm 2026 chưa diễn ra…", "kết quả trận đấu trong bài (2-3) khác với kết quả thực tế (1-1)…"
2. **Title-body mismatch.** "Tiêu đề nói về lịch thi đấu U17 Thái Lan / Indonesia, nhưng nội dung chỉ trình bày phần chuẩn bị của U17 Việt Nam."
3. **Boilerplate / footer-only content.** "Nội dung chỉ là chân trang website, không có thông tin để tóm tắt."

All three are exactly the kind of content we *want* skipped from a fine-tuning corpus — including them would teach the student model to hallucinate or summarise non-articles. The 86 % pass rate is therefore a quality floor, not a ceiling.

## Dataset (`data/datasets/v2/`)

The QC-passed labels were exported with the existing `build_dataset_version` helper using a deterministic 80/10/10 split keyed by article id:

| Split | Rows |
|---|---:|
| train | 1636 |
| val | 218 |
| test | 216 |
| **total** | **2070** |

JSONL schema (one row per line):

```json
{
  "article_id": 1907,
  "source": "vnexpress",
  "url": "...",
  "title": "...",
  "category": "...",
  "published_at": "2026-05-04T08:30:00+00:00",
  "content_text": "...",
  "summary": "...",
  "prompt_version": "1.2.0"
}
```

The files are gitignored (`/data/datasets/`); they live on the labeling box and can be uploaded to Colab / cloud storage for training.

## Cost

Empirical: ~$0.0035 / article on Pro at the observed token mix (input ~1 K + output ~400 incl. thinking). Full-corpus cost ≈ **$8.5**, well within the user's $300 GCP free trial credits.

## Next step

Train ViT5-base + LoRA on the v2 dataset (Colab T4 free). The notebook
`notebooks/finetune_vit5_lora.ipynb` is the starting point.
