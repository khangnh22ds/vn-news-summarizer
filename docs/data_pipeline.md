# Data Pipeline

> Status: outline only — implementation lands in TICKET-002 (Phase 1).

## Stages

1. **Discover** — read `configs/sources.yaml`, fetch each enabled source's
   RSS feeds. Parse with `feedparser`. Yield candidate `(url, title,
   published_at, rss_category)` tuples.
2. **Filter** — drop URLs already present (by `url_hash`). Respect
   `robots.txt` per host (cached for 1h).
3. **Fetch** — `httpx.AsyncClient` with rate limit + retry. Save raw HTML
   to `data/raw/{yyyymmdd}/{url_hash}.html` (optional, gitignored).
4. **Extract** — `trafilatura` → `content_text`, `author`, `published_at`,
   `language`. Fallback to `readability-lxml` on failure.
5. **Normalize** — strip boilerplate, NFC unicode, collapse whitespace,
   trim title/author, normalize publish time to UTC.
6. **Dedupe**:
   - Level 1: canonical URL (strip `utm_*`, `fbclid`, fragment).
   - Level 2: 64-bit SimHash on tokens; Hamming distance ≤ 3 ⇒ duplicate.
7. **Persist** — upsert into `articles` with `status='cleaned'`.
8. **Enqueue** — push article id to inference queue (Redis list / DB
   table) for summarization.

## Schema

See `docs/architecture.md` for the relational schema. Key tables:
`sources`, `articles`, `summaries`, `labels`, `dataset_versions`,
`crawl_runs`, `model_runs`.

## Scheduling

APScheduler in-process for MVP (single worker). Cron-style: every
30 minutes for the discover step, with per-source health metrics:

```text
crawl_runs(source_id, started_at, ended_at, n_new, n_skipped, n_errors, status)
```

Alert (log warning) when `n_errors / (n_new + n_errors) > 0.3` for a run.

## Storage

- **MVP**: SQLite at `./data/app.db`.
- **Switch** to Postgres by changing `DATABASE_URL`. Schema is identical
  thanks to SQLAlchemy 2 + Alembic.
