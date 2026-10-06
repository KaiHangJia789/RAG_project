"""
检索类工具

包含两个:
  - `SearchDocsTool` — 检索知识库（包装 IndexService.search）
  - `WebSearchTool`  — 可插拔的外部搜索；默认用 `LocalCorpusSearch` 实现
    （对已有语料做检索作为"外部搜索"），有 API key 时可替换为真实实现

**为什么 web_search 默认是本地实现**：
  真实搜索 API（Tavily/Serper 等）都需要申请 key。为了不让第 13 周的
  演示被 key 阻塞，这里定义 `SearchBackend` 协议 + 一个不需要 key 的本地
  实现。要接真实 API 时只需实现同一个协议并注入，Agent 侧代码零改动 ——
  这也顺便演示了可插拔设计。
"""
import logging
from typing import Any, Protocol

from app.agent.tools.base import Tool, ToolResult
from app.services.index_service import IndexService

logger = logging.getLogger("rag_api.agent.tools.search")


# ═══════════════════════════════════════════════════════════════
# 工具 1：检索知识库
# ═══════════════════════════════════════════════════════════════

class SearchDocsTool(Tool):
    """
    检索已索引的文档。

    Agent 用它做多跳检索 —— 当首轮检索不理想时，模型可以自己换个说法
    再检索一次（这比第 12 周的"固定重写"更灵活：改写角度由模型决定）。
    """

    name = "search_documents"
    description = (
        "在已上传的知识库文档中做语义检索，返回最相关的文本片段。"
        "当你需要查找事实、定义、步骤等具体信息时使用。"
        "可以用不同的说法多次调用以获取更全面的信息。"
    )

    def __init__(self, index_service: IndexService, *, default_k: int = 5) -> None:
        self.index_service = index_service
        self.default_k = default_k
        self.calls: list[dict] = []      # 调用日志（验收产出要求）

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索查询语句，应当是语义完整的问句或关键词组合",
                },
                "top_k": {
                    "type": "integer",
                    "description": f"返回条数，默认 {self.default_k}",
                    "minimum": 1,
                    "maximum": 20,
                },
            },
            "required": ["query"],
        }

    async def _run(self, query: str = "", top_k: int | None = None, **kwargs: Any) -> ToolResult:
        q = (query or "").strip()
        if not q:
            return ToolResult(ok=False, content="检索查询为空", error="empty_query")

        if not getattr(self.index_service, "ready", False):
            return ToolResult(
                ok=False,
                content="知识库索引尚未就绪，无法检索。",
                error="index_not_ready",
            )

        k = self._clamp_k(top_k)
        hits = await self.index_service.search(q, retrieve_k=20, final_k=k)

        self.calls.append({"query": q, "top_k": k, "hits": len(hits)})

        if not hits:
            return ToolResult(
                ok=True,
                content=f"检索「{q}」没有找到相关内容。",
                meta={"query": q, "hits": 0},
            )

        # 给模型看的文本：带来源标注，便于它引用
        lines = []
        for i, h in enumerate(hits, start=1):
            src = h.filename
            if h.page_number:
                src += f" 第{h.page_number}页"
            lines.append(f"[{i}] 来源：{src}（相似度 {h.score:.3f}）\n{h.text}")

        return ToolResult(
            ok=True,
            content=f"检索「{q}」找到 {len(hits)} 条：\n\n" + "\n---\n".join(lines),
            # 把**原始检索结果**放进 meta，供上层做引用映射。
            # 不要让上层用关键词二次回捞 —— 那样既不准又脆弱。
            meta={"query": q, "hits": len(hits), "max_score": hits[0].score,
                  "raw_hits": hits},
        )

    def _clamp_k(self, top_k: int | None) -> int:
        """钳制 top_k 到合法范围（模型可能给 0、负数或超大值）"""
        try:
            k = int(top_k) if top_k is not None else self.default_k
        except (TypeError, ValueError):
            k = self.default_k
        return max(1, min(20, k))


# ═══════════════════════════════════════════════════════════════
# 工具 2：外部搜索（可插拔后端）
# ═══════════════════════════════════════════════════════════════

class SearchBackend(Protocol):
    """
    搜索后端的协议。

    要接真实的搜索 API（Tavily/Serper/Bing 等），实现这个协议即可 ——
    `WebSearchTool` 不需要任何改动。这就是"可插拔"的含义。
    """

    async def search(self, query: str, *, max_results: int = 5) -> list[dict]:
        """
        Returns:
            [{"title": str, "snippet": str, "url": str}, ...]
        """
        ...


class LocalCorpusSearch:
    """
    本地语料搜索 —— 不需要任何 API key 的默认实现。

    它把**全部已索引文档**当成"外部世界"，用关键词匹配（而非向量检索）
    来找片段。这样：
      - 演示能跑通，不被 key 阻塞
      - 与 `search_documents` 的区别清晰：后者是语义检索、前者是关键词匹配，
        模型可以学到"该用哪个"

    局限：不具备真正的联网能力，只在语料范围内有效。文档里要写明。
    """

    def __init__(self, index_service: IndexService) -> None:
        self.index_service = index_service

    async def search(self, query: str, *, max_results: int = 5) -> list[dict]:
        if not getattr(self.index_service, "ready", False):
            return []

        # 关键词切分（去停用词过重，这里只做简单的字符切分）
        terms = [t for t in self._tokenize(query) if len(t) >= 2]
        if not terms:
            return []

        scored: list[tuple[int, Any]] = []
        for rec in self.index_service.all_records():
            text = rec.text or ""
            score = sum(text.count(t) for t in terms)
            if score > 0:
                scored.append((score, rec))

        scored.sort(key=lambda x: -x[0])

        results = []
        for score, rec in scored[:max_results]:
            snippet = self._snippet(rec.text, terms)
            results.append({
                "title": f"{rec.filename} 第{rec.chunk_index}块",
                "snippet": snippet,
                "url": f"local://{rec.filename}#chunk-{rec.chunk_index}",
                "score": score,
            })
        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """粗粒度分词：英文按空白/标点切，中文按 2-gram 切"""
        import re

        tokens = re.findall(r"[A-Za-z0-9_]+", text)
        chinese = re.findall(r"[一-鿿]+", text)
        for run in chinese:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
            tokens.append(run)
        return tokens

    @staticmethod
    def _snippet(text: str, terms: list[str], *, width: int = 120) -> str:
        """截取包含关键词的片段"""
        for t in terms:
            pos = text.find(t)
            if pos >= 0:
                start = max(0, pos - width // 3)
                return text[start:start + width].strip()
        return text[:width].strip()


class WebSearchTool(Tool):
    """
    外部搜索工具。

    默认后端是 `LocalCorpusSearch`（本地语料关键词匹配，无需 key）。
    注入真实后端即可获得联网能力，本类无需改动。
    """

    name = "web_search"
    description = (
        "在知识库之外的资料中搜索信息。"
        "当知识库检索不到、或问题需要外部/最新信息时使用。"
        "注意：结果可能不如知识库精确，优先尝试 search_documents。"
    )

    def __init__(self, backend: SearchBackend) -> None:
        self.backend = backend
        self.calls: list[dict] = []

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "max_results": {
                    "type": "integer",
                    "description": "返回条数，默认 5",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        }

    async def _run(self, query: str = "", max_results: int | None = None, **kwargs: Any) -> ToolResult:
        q = (query or "").strip()
        if not q:
            return ToolResult(ok=False, content="搜索关键词为空", error="empty_query")

        try:
            n = int(max_results) if max_results is not None else 5
        except (TypeError, ValueError):
            n = 5
        n = max(1, min(10, n))

        results = await self.backend.search(q, max_results=n)
        self.calls.append({"query": q, "max_results": n, "results": len(results)})

        if not results:
            return ToolResult(
                ok=True, content=f"搜索「{q}」没有找到结果。", meta={"results": 0}
            )

        lines = [
            f"{i}. {r.get('title', '')}\n   {r.get('snippet', '')}"
            for i, r in enumerate(results, start=1)
        ]
        return ToolResult(
            ok=True,
            content=f"搜索「{q}」找到 {len(results)} 条：\n\n" + "\n".join(lines),
            meta={"results": len(results), "backend": type(self.backend).__name__},
        )
