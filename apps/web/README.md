# apps/web — Next.js frontend (placeholder)

This directory is intentionally empty until **Phase 5 / TICKET-007**.

When ready, scaffold here with:

```bash
cd apps/web
npx create-next-app@latest . --ts --tailwind --eslint --app --src-dir --import-alias "@/*"
```

Then:

```bash
npx shadcn@latest init
```

The frontend will consume the FastAPI backend at `NEXT_PUBLIC_API_BASE_URL`
(see `.env.example`).
