export type ApiResponse<T> = {
  code: number;
  message: string;
  data: T;
};

export type Paginated<T> = {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
};

export type Document = {
  id: string;
  filename: string;
  file_type: string;
  file_size: number;
  status: string;
  created_at: string;
};

export type Citation = {
  filename: string;
  chunk_index: number;
  score: number;
  text: string;
  page_number: number | null;
};

export type RagAnswer = {
  question: string;
  answer: string;
  citations: Citation[];
  refused: boolean;
  refusal_reason: string | null;
  latency_ms: number;
  trace_url: string | null;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api/v1${path}`, init);
  if (!res.ok) {
    throw new Error(`请求失败：${res.status}`);
  }
  const body = (await res.json()) as ApiResponse<T>;
  return body.data;
}

export function listDocuments(page = 1, pageSize = 20) {
  return request<Paginated<Document>>(
    `/documents?page=${page}&page_size=${pageSize}`,
  );
}

export function deleteDocument(id: string) {
  return request<unknown>(`/documents/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
}

export function askQuestion(question: string) {
  return request<RagAnswer>("/qa/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
}
