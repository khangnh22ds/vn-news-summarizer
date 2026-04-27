# vn-news-summarizer

> Vietnamese real-time news summarization — research/educational project.
> Crawl Vietnamese news, label summaries with an LLM teacher (Vertex AI
> Gemini), fine-tune a smaller model (`VietAI/vit5-base`), and serve
> daily summaries through a FastAPI + Next.js web app.

**Status:** Phase 0 complete (this scaffold). Phase 1 (crawler + DB) is
the next ticket. See [`docs/roadmap.md`](docs/roadmap.md).

---

## Why this project

- **Educational.** Hands-on with Vietnamese NLP, LLM-as-teacher labeling,
  fine-tuning seq2seq, FastAPI, Next.js, and a real data pipeline.
- **Practical.** Aim for a small repo you can clone and run locally in
  ≤ 15 minutes, not a microservice empire.
- **Honest about copyright.** We summarize short, link to the original,
  and respect `robots.txt`. See [`docs/legal_notes.md`](docs/legal_notes.md).

## Repository layout

```
vn-news-summarizer/
├── apps/
│   └── web/                 # Next.js frontend (Phase 5)
├── configs/
│   ├── sources.yaml         # 8 Vietnamese news sources, RSS-first
│   ├── prompts/             # Versioned LLM prompts
│   └── training/            # Versioned training configs
├── data/                    # gitignored, DVC later
├── docs/                    # architecture, legal, prompt, training, deployment
├── models/                  # gitignored model checkpoints
├── notebooks/               # EDA & error analysis
├── packages/
│   ├── common/              # ORM, schemas, settings
│   ├── crawler/             # RSS + extract + dedupe
│   ├── labeling/            # Vertex AI labeling + QC
│   ├── training/            # Baselines + fine-tune + eval
│   ├── inference/           # Serve model + LLM fallback
│   └── api/                 # FastAPI app
├── scripts/                 # CLI entrypoints (run_crawler.py, ...)
├── secrets/                 # gitignored (place vertex-sa.json here)
├── tests/                   # unit + integration
├── .env.example
├── docker-compose.yml
├── Makefile
└── pyproject.toml           # uv workspace root
```

## Tech stack (MVP)

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11 | uv workspace |
| Crawler | httpx + feedparser + trafilatura + APScheduler | RSS first |
| LLM teacher | **Vertex AI** (Gemini 2.0 Flash) | `google-cloud-aiplatform` |
| Vietnamese tokenize | pyvi + underthesea | for ROUGE/QC |
| Training | transformers + PEFT (LoRA) + MLflow | Colab/Kaggle T4 |
| DB | SQLite (MVP) → Postgres (later) | SQLAlchemy 2 + Alembic |
| Backend | FastAPI + Pydantic v2 | Uvicorn |
| Frontend | Next.js 14 + Tailwind + shadcn/ui | App Router |
| Container | Docker + docker-compose | Local-only for MVP |
| Lint/format/types | ruff + mypy strict | one config |

## Prerequisites

- **Python 3.11** (managed via uv).
- **uv** ≥ 0.7.x — install with
  `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- **Docker** + Docker Compose v2 — for `db` and `redis` services.
- **Node.js** ≥ 20 — only needed for `apps/web` from Phase 5 onward.
- **Google Cloud project** with **Vertex AI** enabled and a service
  account JSON with the `Vertex AI User` role (free tier covers MVP
  labeling). Place the JSON at `./secrets/vertex-sa.json`.

## Quick start

```bash
git clone https://github.com/khangnh22ds/vn-news-summarizer.git
cd vn-news-summarizer

cp .env.example .env             # then edit values
make setup                       # uv sync workspace + dev deps
make db-up                       # docker compose up -d db redis (optional for MVP)
make lint                        # ruff + mypy
make test                        # pytest smoke tests
make api                         # FastAPI on http://localhost:8000  (try /healthz, /docs)
```

Phase-specific commands light up as the corresponding ticket lands:

```bash
make crawl       # Phase 1
make label       # Phase 2
make eval        # Phase 3
make train       # Phase 4
make web         # Phase 5
```

See [`docs/roadmap.md`](docs/roadmap.md) for what each phase delivers.

## Configuration

All runtime config goes through `.env` (see `.env.example`).
Source list and prompts are in `configs/`:
- [`configs/sources.yaml`](configs/sources.yaml) — 8 Vietnamese news sources.
- [`configs/prompts/summarize_v1.yaml`](configs/prompts/summarize_v1.yaml) —
  versioned LLM prompt + QC thresholds.
- [`configs/training/vit5_base_v1.yaml`](configs/training/vit5_base_v1.yaml) —
  versioned fine-tune config.

## Development

- **Pre-commit** (recommended): `make precommit`. Runs ruff + ruff-format
  + mypy + basic hygiene on each commit.
- **Format on save**: configure your editor for ruff.
- **Add a workspace package**: drop a new directory under `packages/`
  with its own `pyproject.toml`, then add it to the root `pyproject.toml`
  `[tool.uv.sources]` and `[project] dependencies` lists.

## Contributing

This is a personal/educational project, but issues and PRs are welcome.
For substantial changes please open an issue first to discuss.

## Legal & ethical

Read [`docs/legal_notes.md`](docs/legal_notes.md) before crawling. tl;dr:
- RSS-first; respect robots.txt; ≤ 1 req/sec/host.
- Short summaries (≤ 70 words) with strong source attribution + link-out.
- Project is **non-commercial / research only**. Takedown contact is
  the maintainer email in `LICENSE`.

## License

MIT — see [`LICENSE`](LICENSE). Crawled news content remains the
property of its respective publishers.
