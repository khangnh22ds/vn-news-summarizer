# Architecture

> Status: TICKET-001 (Phase 0). Subject to refinement as we ship Phase 1+.

## High-level

```
                    ┌──────────────────────┐
                    │  Frontend (Next.js)  │
                    │  - SSR list bài      │
                    │  - revalidate 5 min  │
                    └─────────┬────────────┘
                              │ HTTPS
                    ┌─────────▼────────────┐
                    │  Backend API         │   ┌────────────┐
                    │  FastAPI + Uvicorn   │◄──┤   Redis    │ cache (optional)
                    │  /api/articles       │   └────────────┘
                    │  /api/admin/*        │
                    └─┬─────────────┬──────┘
                      │             │
              ┌───────▼──────┐  ┌───▼──────────────┐
              │  PostgreSQL  │  │ Inference svc    │
              │  (or SQLite) │  │ ViT5 + LLM       │
              └──────▲───────┘  │ fallback         │
                     │          └──────────────────┘
                     │
            ┌────────┴────────────────────────┐
            │                                 │
   ┌────────▼─────────┐               ┌───────▼────────┐
   │ Crawler worker   │               │ Labeling worker│
   │ APScheduler      │               │ Vertex AI      │
   │  - RSS fetch     │               │ + QC pipeline  │
   │  - extract       │               └────────────────┘
   │  - dedupe        │
   └──────────────────┘
```

## Three planes

1. **Data plane** — crawler + DB. Independent of any ML. Owns
   `articles`, `sources`, `crawl_runs`.
2. **ML plane** — labeling, training, evaluation. Reads `articles`,
   writes `labels`, `summaries`, `dataset_versions`, `model_runs`.
3. **Serving plane** — API + web + cache. Reads from DB, calls inference
   service. Writes only via admin endpoints.

## Components

| Component | Tech | MVP role |
|---|---|---|
| Crawler | httpx + feedparser + trafilatura + APScheduler | Pull RSS every 30 min, extract text, dedupe, persist. |
| DB | SQLite (MVP) → Postgres (later) | Source of truth for articles, summaries, labels, runs. |
| Labeling worker | google-cloud-aiplatform (Vertex AI Gemini) | Generate teacher summaries with QC. Batch nightly. |
| Training | transformers + PEFT (LoRA) + MLflow | Fine-tune ViT5-base on labeled data, log to MLflow. |
| Inference | transformers + LLM fallback | Serve summaries; fall back to Vertex AI if QC fails. |
| API | FastAPI + Pydantic v2 + SQLAlchemy 2 | Public read API + admin endpoints. |
| Frontend | Next.js 14 (App Router) + Tailwind + shadcn/ui | Public list, filters, source attribution. |
| Cache | Redis (optional) | Cache hot list endpoints with TTL. |
| Observability | loguru + prometheus_client + Sentry (optional) | Structured logs + /metrics + error tracking. |

## Daily refresh strategy

- Crawler: every 30 min (cron via APScheduler).
- Inference enqueued per-article on insert. Inference workers pull queue.
- Frontend: ISR revalidate every 5 min on the list page.
- LLM labeling: **off the hot path** — runs nightly to grow training set;
  not required for serving.

## Why monorepo?

- One shared package (`vn-news-common`) for ORM models + Pydantic schemas
  used by the crawler, labeler, training, inference, and API.
- Single CI, single Makefile, single `uv sync`.
- Frontend lives at `apps/web/` and is independent of Python.

## Failure modes & fallbacks

| Failure | Detection | Fallback |
|---|---|---|
| Source RSS broken | Per-source health metric (parse rate < threshold) | Disable source, alert via log |
| Trafilatura fails | Empty `content_text` | Mark `status='failed'`, retry with readability-lxml |
| Model summary fails QC (entity/number) | QC layer in inference | Call Vertex AI fallback, cache result |
| Vertex AI down | Tenacity retries exhausted | Serve extractive baseline (TextRank) |
| DB down | API healthcheck | 503 with retry-after; web shows cached list |
