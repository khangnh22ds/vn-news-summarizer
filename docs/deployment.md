# Deployment

> Scope per user decision: **local Docker only for MVP.** Phase 6
> deployment is deferred. This document captures the local-first plan
> and a forward-looking option for when remote hosting is wanted.

## Local Docker (target for Phase 6)

```bash
cp .env.example .env          # fill in Vertex AI creds + ADMIN_TOKEN
make setup                    # uv sync workspace
make db-up                    # postgres + redis
make db-migrate               # once Phase 1 lands
make api                      # FastAPI on :8000
cd apps/web && npm run dev    # Next.js on :3000 (after Phase 5)
```

When all services are wired up:

```bash
make up      # docker compose up -d --build
make logs    # follow logs
make down
```

## Cron / scheduled crawl on a single host

The crawler runs in-process via APScheduler (every 30 min). For local
development, just keep `make crawl` or the `crawler` docker service
running. No external scheduler needed.

If you ever move off a single host, switch to:
- Celery beat + Redis (multi-worker), or
- A small VPS cron entry (`*/30 * * * *`) running `make crawl`.

## Future: remote hosting (not in scope for MVP)

- Backend → Fly.io free tier (1 shared-cpu-1x VM is enough for a
  research-scale workload).
- Frontend → Vercel free hobby tier (auto SSR + ISR).
- DB → Fly Postgres or Supabase free tier.
- Cron → GitHub Actions schedule trigger or Fly Machines.

When ready, file a new ticket — Phase 6 — to add `fly.toml`, configure
secrets in the providers, and rewire `docker-compose.yml`.
