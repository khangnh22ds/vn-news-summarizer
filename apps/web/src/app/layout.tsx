import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "vn-news-summarizer",
  description:
    "Tóm tắt tin tức tiếng Việt bằng ViT5-base + LoRA (mục đích nghiên cứu / giáo dục).",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="vi">
      <body className="antialiased min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
        {children}
      </body>
    </html>
  );
}
