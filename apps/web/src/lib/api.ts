/**
 * Client-side helpers for talking to the FastAPI backend.
 *
 * The base URL comes from `NEXT_PUBLIC_API_BASE_URL` (defaults to
 * http://localhost:8000 for local dev). All call-sites should go
 * through `summarize()` so the request/response shape stays in lockstep
 * with `packages/api/src/vn_news_api/routers/summarize.py`.
 */

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export type SummarizeRequest =
  | { text: string; url?: never }
  | { url: string; text?: never };

export interface SummarizeResponse {
  summary: string;
  model_id: string;
  source_url: string | null;
  source_title: string | null;
  input_chars: number;
  summary_chars: number;
  elapsed_ms: number;
}

export interface SummarizeErrorBody {
  detail:
    | string
    | { reason: string; message: string }
    | Array<{ loc: (string | number)[]; msg: string; type: string }>;
}

export class SummarizeError extends Error {
  status: number;
  reason?: string;

  constructor(message: string, status: number, reason?: string) {
    super(message);
    this.name = "SummarizeError";
    this.status = status;
    this.reason = reason;
  }
}

export async function summarize(
  payload: SummarizeRequest,
  signal?: AbortSignal,
): Promise<SummarizeResponse> {
  const res = await fetch(`${API_BASE_URL}/summarize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });

  if (!res.ok) {
    let reason: string | undefined;
    let message = `HTTP ${res.status}`;
    try {
      const body = (await res.json()) as SummarizeErrorBody;
      const d = body.detail;
      if (typeof d === "string") {
        message = d;
      } else if (Array.isArray(d)) {
        message = d.map((e) => `${e.loc.join(".")}: ${e.msg}`).join("; ");
      } else if (d && typeof d === "object") {
        reason = d.reason;
        message = d.message;
      }
    } catch {
      /* ignore JSON parse errors and fall through with HTTP status */
    }
    throw new SummarizeError(message, res.status, reason);
  }

  return (await res.json()) as SummarizeResponse;
}
