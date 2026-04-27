# Roadmap

Full-time pace target: ~3 weeks total. Each phase ends with a green PR
and a runnable demo.

| Phase | Title | Days | Status |
|---|---|---|---|
| 0 | Research & repo scaffolding (TICKET-001) | 1–2 | **DONE** |
| 1 | Crawler + DB (TICKET-002) | 3 | next |
| 2 | LLM labeling + dataset v1 (TICKET-003) | 4 | |
| 3 | Baselines + eval harness (TICKET-004) | 2 | |
| 4 | Fine-tune ViT5-base (TICKET-005) | 5 | |
| 5 | Inference service + API + Next.js web (TICKET-006/007) | 5 | |
| 6 | Human eval + local Docker hardening (TICKET-008) | 2 | |

See `docs/architecture.md` and the individual ticket descriptions
(filed per phase) for the breakdown of acceptance criteria.

## Outstanding decisions

- Bump LLM teacher model to `gemini-1.5-pro` for ~10% of label set as a
  diversity check — defer until QC pass-rate stats are available.
- Whether to publish the gold dataset for research re-use — defer to
  Phase 6 review.
