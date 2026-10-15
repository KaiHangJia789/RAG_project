import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG 知识库",
  description: "文档问答与引用溯源",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
