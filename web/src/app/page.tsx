"use client";

import { FormEvent, useEffect, useState } from "react";
import {
  askQuestion,
  deleteDocument,
  listDocuments,
  type Document,
  type RagAnswer,
} from "@/lib/api";

export default function HomePage() {
  const [docs, setDocs] = useState<Document[]>([]);
  const [docError, setDocError] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<RagAnswer | null>(null);
  const [asking, setAsking] = useState(false);
  const [qaError, setQaError] = useState<string | null>(null);

  async function refreshDocs() {
    try {
      const data = await listDocuments();
      setDocs(data.items);
      setDocError(null);
    } catch (e) {
      setDocError(e instanceof Error ? e.message : "加载失败");
    }
  }

  useEffect(() => {
    refreshDocs();
  }, []);

  async function handleDelete(id: string) {
    if (!window.confirm("确认删除该文档？")) return;
    try {
      await deleteDocument(id);
      await refreshDocs();
    } catch (e) {
      setDocError(e instanceof Error ? e.message : "删除失败");
    }
  }

  async function handleAsk(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const q = question.trim();
    if (!q) return;
    setAsking(true);
    setQaError(null);
    try {
      setAnswer(await askQuestion(q));
    } catch (err) {
      setAnswer(null);
      setQaError(err instanceof Error ? err.message : "问答失败");
    } finally {
      setAsking(false);
    }
  }

  return (
    <main style={{ maxWidth: 900, margin: "0 auto", padding: 24 }}>
      <h1>RAG 知识库</h1>

      <section>
        <h2>文档列表</h2>
        {docError && <p role="alert">{docError}</p>}
        {docs.length === 0 && !docError && <p>暂无文档</p>}
        <ul>
          {docs.map((d) => (
            <li key={d.id}>
              {d.filename}（{d.status}）{" "}
              <button onClick={() => handleDelete(d.id)}>删除</button>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h2>问答</h2>
        <form onSubmit={handleAsk}>
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="输入问题"
            maxLength={2000}
            style={{ width: "70%" }}
          />{" "}
          <button type="submit" disabled={asking}>
            {asking ? "生成中…" : "提问"}
          </button>
        </form>
        {qaError && <p role="alert">{qaError}</p>}
        {answer && (
          <div>
            <p>
              {answer.refused
                ? `拒答：${answer.refusal_reason ?? ""}`
                : answer.answer}
            </p>
            {answer.citations.length > 0 && (
              <ol>
                {answer.citations.map((c, i) => (
                  <li key={`${c.filename}-${c.chunk_index}-${i}`}>
                    {c.filename}
                    {c.page_number != null && `（第 ${c.page_number} 页）`}
                    ：{c.text.slice(0, 120)}
                  </li>
                ))}
              </ol>
            )}
            {answer.trace_url && (
              <a href={answer.trace_url} target="_blank" rel="noreferrer">
                查看 LangFuse Trace
              </a>
            )}
          </div>
        )}
      </section>
    </main>
  );
}
