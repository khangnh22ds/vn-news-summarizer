"use client";

import { useState } from "react";
import {
  API_BASE_URL,
  SummarizeError,
  type SummarizeResponse,
  summarize,
} from "@/lib/api";

type Mode = "url" | "text";

export default function Page() {
  const [mode, setMode] = useState<Mode>("url");
  const [url, setUrl] = useState("");
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SummarizeResponse | null>(null);

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const payload =
        mode === "url"
          ? { url: url.trim() }
          : { text: text.trim() };
      if (mode === "url" && !payload.url) {
        throw new Error("Vui lòng nhập URL bài báo.");
      }
      if (mode === "text" && !payload.text) {
        throw new Error("Vui lòng nhập nội dung bài báo.");
      }
      const res = await summarize(payload);
      setResult(res);
    } catch (err) {
      if (err instanceof SummarizeError) {
        setError(
          err.reason ? `${err.reason}: ${err.message}` : err.message,
        );
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Lỗi không xác định.");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="mx-auto max-w-3xl px-4 py-10">
      <header className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">
          vn-news-summarizer
        </h1>
        <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">
          Tóm tắt tin tức tiếng Việt bằng ViT5-base + LoRA. Nhập URL hoặc dán
          nội dung bài báo. Backend:{" "}
          <code className="font-mono text-xs">{API_BASE_URL}</code>
        </p>
      </header>

      <form onSubmit={onSubmit} className="space-y-4">
        <div
          className="inline-flex rounded-md border border-slate-300 dark:border-slate-700 p-1 text-sm"
          role="tablist"
        >
          <button
            type="button"
            role="tab"
            aria-selected={mode === "url"}
            onClick={() => setMode("url")}
            className={`px-3 py-1 rounded ${
              mode === "url"
                ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900"
                : "text-slate-700 dark:text-slate-300"
            }`}
          >
            URL bài báo
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === "text"}
            onClick={() => setMode("text")}
            className={`px-3 py-1 rounded ${
              mode === "text"
                ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900"
                : "text-slate-700 dark:text-slate-300"
            }`}
          >
            Văn bản trực tiếp
          </button>
        </div>

        {mode === "url" ? (
          <div>
            <label htmlFor="url" className="block text-sm font-medium mb-1">
              URL
            </label>
            <input
              id="url"
              type="url"
              required
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://vnexpress.net/..."
              className="w-full rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-slate-500"
            />
          </div>
        ) : (
          <div>
            <label htmlFor="text" className="block text-sm font-medium mb-1">
              Nội dung bài báo
            </label>
            <textarea
              id="text"
              required
              rows={10}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Dán nội dung bài báo vào đây…"
              className="w-full rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-slate-500"
            />
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-50 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-slate-300"
        >
          {loading ? "Đang tóm tắt…" : "Tóm tắt"}
        </button>
      </form>

      {error && (
        <div
          className="mt-6 rounded-md border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-800 dark:bg-red-950 dark:text-red-200"
          role="alert"
        >
          {error}
        </div>
      )}

      {result && (
        <section className="mt-6 rounded-md border border-slate-300 bg-white p-5 shadow-sm dark:border-slate-700 dark:bg-slate-900">
          {result.source_title && (
            <h2 className="text-lg font-semibold mb-2">{result.source_title}</h2>
          )}
          <p className="whitespace-pre-wrap text-sm leading-6">
            {result.summary}
          </p>
          <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-slate-600 dark:text-slate-400 sm:grid-cols-4">
            <dt className="font-medium">Model</dt>
            <dd className="font-mono col-span-1 sm:col-span-3 truncate">
              {result.model_id}
            </dd>
            <dt className="font-medium">Input</dt>
            <dd>{result.input_chars} ký tự</dd>
            <dt className="font-medium">Tóm tắt</dt>
            <dd>{result.summary_chars} ký tự</dd>
            <dt className="font-medium">Thời gian</dt>
            <dd>{result.elapsed_ms} ms</dd>
            {result.source_url && (
              <>
                <dt className="font-medium">Nguồn</dt>
                <dd className="col-span-1 sm:col-span-3 truncate">
                  <a
                    href={result.source_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="underline hover:text-slate-900 dark:hover:text-slate-100"
                  >
                    {result.source_url}
                  </a>
                </dd>
              </>
            )}
          </dl>
        </section>
      )}

      <footer className="mt-10 border-t border-slate-200 pt-4 text-xs text-slate-500 dark:border-slate-800 dark:text-slate-400">
        Mục đích nghiên cứu / giáo dục. Tuân thủ điều khoản nguồn tin và{" "}
        <a
          href="https://github.com/khangnh22ds/vn-news-summarizer"
          className="underline"
          target="_blank"
          rel="noopener noreferrer"
        >
          repo trên GitHub
        </a>
        .
      </footer>
    </main>
  );
}
