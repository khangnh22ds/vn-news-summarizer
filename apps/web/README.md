# apps/web — vn-news-summarizer frontend

Next.js 14 (App Router, TypeScript, Tailwind) UI for the FastAPI backend
in [`packages/api`](../../packages/api). Single page with a URL/text
toggle that POSTs to `/summarize` and renders the model output plus
metadata (model id, char counts, latency).

## Local dev

```bash
# 1. install deps (one-off)
cd apps/web && npm install

# 2. point at the backend (defaults to http://localhost:8000)
cp .env.example .env.local

# 3. run
npm run dev          # http://localhost:3000
```

In a second terminal start the backend:

```bash
make api             # uvicorn vn_news_api.app:app --reload --port 8000
```

For smoke tests against an HF Hub-hosted adapter you need
`HF_TOKEN` exported (the default `MODEL_PATH` is the private
`Gthgfuiss123/vit5-news-vi-lora-v2` repo). To skip the model download
entirely, set `MODEL_PATH=VietAI/vit5-base` in `.env`.

## Build / lint

```bash
npm run lint
npm run build
```
