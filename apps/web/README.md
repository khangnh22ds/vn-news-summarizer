# apps/web — vn-news-summarizer frontend

Next.js 14 (App Router, TypeScript, Tailwind) UI for the FastAPI backend
in [`packages/api`](../../packages/api). Single page with a URL/text
toggle that POSTs to `/summarize` and renders the model output plus
metadata (model id, char counts, latency).

This app is a **standalone Node.js project** — it has its own
`package.json` and does not depend on the Python tooling (uv / Makefile)
in the rest of the monorepo. You can develop the frontend in isolation
without ever touching `packages/`.

## Local dev

```bash
# 1. install deps (one-off)
cd apps/web && npm install

# 2. point at the backend (defaults to http://localhost:8000)
cp .env.example .env.local

# 3. run
npm run dev          # http://localhost:3000
```

In a second terminal start the backend (from the repo root):

```bash
# one-off: extract the LoRA tarball into ./models/
make bootstrap-model TARBALL=path/to/vit5-news-v2.tar.gz

# then:
make api             # uvicorn vn_news_api.app:app --reload --port 8000
```

The default `MODEL_PATH=./models/vit5-news-v2` means the backend reads
the adapter straight off disk — no Hugging Face token required. If you
want to pull the adapter from the Hub instead, set
`MODEL_PATH=Gthgfuiss123/vit5-news-vi-lora-v2` and `HF_TOKEN=...` in
`.env` at the repo root.

## Build / lint

```bash
npm run lint
npm run build
```
