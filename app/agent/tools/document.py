"""
文档读取工具 —— 按 ID 取文档元信息与分块正文

**为什么需要它**：
  `search_documents` 只能按语义检索到零散片段，无法回答「这份文档到底讲了什么」
  「第 3 章的内容是什么」这类需要通读的问题。这个工具补上那一块。

**放在 `app/agent/tools/` 而不是 MCP 目录下**：两个消费方都要用它 ——
  - LangGraph Agent（`app/agent/`）
  - MCP Server（`app/mcp_server/`）
  所以它属于共用的工具层。

**两个关键设计**：

1. **分页是必需的，不是优化**。实测 `student_hardbook.pdf` 有 325 个分块、
   中位长度 575 字符、总计约 19 万字符。一次返回全文会直接撑爆上下文窗口。
   默认 20 块（约 11.5k 字符 ≈ 4-6k token），硬上限 50 块。

2. **必须按 chunk_strategy 过滤**。005 迁移后的唯一约束是
   `(document_id, chunk_strategy, chunk_index)`，同一文档可以有多套策略的分块共存。
   不过滤就会拿到交错重复的行（同一块在不同策略下各出现一次）。
"""
import logging
from typing import Any

import asyncpg

from app.agent.tools.base import Tool, ToolResult
from app.config import settings

logger = logging.getLogger("rag_api.agent.tools.document")

# 分页默认值与上限（依据真实数据规模定，不是拍脑袋）
DEFAULT_LIMIT = 20
MAX_LIMIT = 50


class GetDocumentTool(Tool):
    """
    按 ID 读取文档的元信息与分块正文（分页）。

    只依赖两个 Repository —— **不需要 Redis / DocumentService / StorageService**：
      - 元信息：`DocumentRepository.get_by_id()`
      - 正文：`ChunkRepository.fetch_by_document()`

    这样做的收益：MCP Server 作为独立进程时不必装配缓存层，
    也避开了 `StorageService.__init__` 里那个依赖工作目录的 `mkdir` 副作用。
    """

    name = "get_document_by_id"
    description = (
        "按文档 ID 读取文档的元信息与分块正文，支持分页。"
        "用于回答「这份文档讲了什么」「文档有多少内容」这类需要通读的问题。"
        "注意：大文档必须分页读取，返回结果末尾会提示如何取下一页。"
    )

    def __init__(
        self,
        db,
        *,
        doc_repo=None,
        chunk_repo=None,
        strategy: str | None = None,
        default_limit: int = DEFAULT_LIMIT,
        max_limit: int = MAX_LIMIT,
    ) -> None:
        """
        Args:
            db: `Database` 实例（用它的连接池）
            doc_repo / chunk_repo: 可注入，便于测试替换
            strategy: 分块策略，None 时取 `settings.CHUNK_STRATEGY`
            default_limit / max_limit: 分页参数
        """
        if default_limit <= 0:
            raise ValueError(f"default_limit 必须 > 0，实际 {default_limit}")
        if max_limit < default_limit:
            raise ValueError(
                f"max_limit({max_limit}) 不能小于 default_limit({default_limit})"
            )

        self.db = db
        self.strategy = strategy or settings.CHUNK_STRATEGY
        self.default_limit = default_limit
        self.max_limit = max_limit
        self.calls: list[dict] = []      # 调用日志（与其它工具保持一致）

        # 延迟导入避免循环依赖（repositories 不依赖 tools，但保持一致的注入风格）
        if doc_repo is None or chunk_repo is None:
            from app.db.repositories.chunk_repo import ChunkRepository
            from app.db.repositories.document_repo import DocumentRepository

            doc_repo = doc_repo or DocumentRepository()
            chunk_repo = chunk_repo or ChunkRepository()
        self.doc_repo = doc_repo
        self.chunk_repo = chunk_repo

    # ═══════════════════════════════════════════════════════════
    # schema（供 LangGraph Agent 用；MCP 那边从类型注解生成，不用这个）
    # ═══════════════════════════════════════════════════════════

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "文档 UUID（可从文档列表中获得）",
                },
                "offset": {
                    "type": "integer",
                    "description": "起始分块序号，从 0 开始，默认 0",
                    "minimum": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": f"返回分块数，默认 {DEFAULT_LIMIT}，上限 {MAX_LIMIT}",
                    "minimum": 1,
                    "maximum": MAX_LIMIT,
                },
            },
            "required": ["document_id"],
        }

    # ═══════════════════════════════════════════════════════════
    # 主逻辑
    # ═══════════════════════════════════════════════════════════

    async def _run(
        self,
        document_id: str = "",
        offset: int = 0,
        limit: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        doc_id = (document_id or "").strip()
        if not doc_id:
            return ToolResult(
                ok=False, content="必须提供 document_id", error="missing_id"
            )

        off = self._clamp_offset(offset)
        lim = self._clamp_limit(limit)

        self.calls.append({"document_id": doc_id, "offset": off, "limit": lim})

        if self.db is None or getattr(self.db, "pool", None) is None:
            return ToolResult(
                ok=False,
                content="数据库不可用，无法读取文档。请确认 PostgreSQL 已启动。",
                error="db_unavailable",
            )

        try:
            async with self.db.pool.acquire() as conn:
                doc_row = await self._fetch_document(conn, doc_id)
                if doc_row is None:
                    return ToolResult(
                        ok=False,
                        content=(
                            f"未找到 ID 为 '{doc_id}' 的文档。"
                            f"请确认 ID 是否正确（可从文档列表获取）。"
                        ),
                        error="document_not_found",
                    )

                chunks, used_strategy = await self._fetch_chunks(conn, doc_id)

        except asyncpg.PostgresError as e:
            logger.error("查询文档失败 [%s]: %s", doc_id, e)
            return ToolResult(
                ok=False, content=f"数据库查询失败：{e}", error="db_error"
            )

        if not chunks and off > 0:
            return ToolResult(
                ok=False,
                content=f"offset={off} 超出范围（本文档共 0 个分块）。",
                error="offset_out_of_range",
            )

        return ToolResult(
            ok=True,
            content=self._format(doc_row, chunks, off, lim, used_strategy),
            meta={
                "document_id": doc_id,
                "filename": doc_row.get("filename"),
                "total_chunks": len(chunks),
                "offset": off,
                "limit": lim,
                "strategy": used_strategy,
            },
        )

    # ═══════════════════════════════════════════════════════════
    # 数据获取
    # ═══════════════════════════════════════════════════════════

    async def _fetch_document(self, conn, doc_id: str):
        """取文档元信息。ID 非 UUID 时 asyncpg 会报类型错，这里转成未找到。"""
        try:
            row = await self.doc_repo.get_by_id(conn, doc_id)
        except (asyncpg.DataError, asyncpg.PostgresError) as e:
            # 'abc' 这种非 UUID 字符串会触发 invalid input syntax for type uuid
            logger.warning("文档 ID 格式非法 [%s]: %s", doc_id, e)
            return None
        return dict(row) if row else None

    async def _fetch_chunks(
        self, conn, doc_id: str
    ) -> tuple[list[dict], str]:
        """
        取该文档的分块（按 chunk_index 升序）。

        **先按当前策略过滤**：同一文档可能有多套策略的分块共存
        （唯一约束含 chunk_strategy），不过滤会拿到交错重复的行。

        若当前策略下无数据，回退为不过滤并如实报告 —— 避免因为
        「DB 是用别的策略建的」而让工具完全不能用。
        """
        rows = await self.chunk_repo.fetch_by_document(conn, doc_id, self.strategy)
        if rows:
            return [dict(r) for r in rows], self.strategy

        rows = await self.chunk_repo.fetch_by_document(conn, doc_id, None)
        if rows:
            strategies = sorted({r["chunk_strategy"] for r in rows})
            used = "+".join(strategies)
            logger.info(
                "文档 %s 在策略 '%s' 下无分块，回退为全部策略（%s）",
                doc_id, self.strategy, used,
            )
            # 多策略共存时按 (strategy, chunk_index) 排序，保证输出稳定
            ordered = sorted(rows, key=lambda r: (r["chunk_strategy"], r["chunk_index"]))
            return [dict(r) for r in ordered], used

        return [], self.strategy

    # ═══════════════════════════════════════════════════════════
    # 输出格式化
    # ═══════════════════════════════════════════════════════════

    def _format(
        self,
        doc: dict,
        all_chunks: list[dict],
        offset: int,
        limit: int,
        strategy: str,
    ) -> str:
        """
        组装给模型看的文本。

        格式目标是让模型知道三件事：① 这是哪份文档 ② 当前读到哪儿了
        ③ 怎么继续读下一页。缺任何一项都会导致模型要么重复请求、
        要么以为已经读完了。
        """
        total = len(all_chunks)
        page = all_chunks[offset:offset + limit]
        start = offset + 1 if page else 0
        end = offset + len(page)
        remaining = max(0, total - end)

        # 「剩余 0 块」是噪音，读完了就不显示 —— 让模型只关注真正有用的信息
        position = f"当前：第 {start}–{end} 块（offset={offset}, limit={limit}"
        position += f"，剩余 {remaining} 块）" if remaining else "）"

        lines = [
            f"文档：{doc.get('filename', '(未知)')}",
            f"ID：{doc.get('id')}  类型：{doc.get('file_type', '?')}  "
            f"大小：{self._human_size(doc.get('file_size'))}  "
            f"状态：{doc.get('status', '?')}",
            f"分块总数：{total}（策略 {strategy}）  {position}",
            "",
        ]

        if not page:
            lines.append("（该范围内没有分块）")
        else:
            for c in page:
                idx = c.get("chunk_index", 0)
                page_no = c.get("page_number")
                header = f"[块 {idx}" + (f" · 第 {page_no} 页]" if page_no else "]")
                lines.append(header)
                lines.append((c.get("chunk_text") or "").strip())
                lines.append("")

            # 续取提示 —— 让模型知道还有内容且知道怎么拿
            if remaining > 0:
                lines.append(
                    f"（还有 {remaining} 块未显示。"
                    f'继续请调用 get_document_by_id(document_id="{doc.get("id")}", '
                    f"offset={end}, limit={limit})）"
                )
            else:
                lines.append("（已显示全部内容）")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════
    # 参数钳制（防御模型给的越界值）
    # ═══════════════════════════════════════════════════════════

    def _clamp_limit(self, limit: int | None) -> int:
        """钳制 limit 到 [1, max_limit]。非法值回落默认值。"""
        try:
            n = int(limit) if limit is not None else self.default_limit
        except (TypeError, ValueError):
            return self.default_limit
        return max(1, min(self.max_limit, n))

    @staticmethod
    def _clamp_offset(offset: int | None) -> int:
        try:
            n = int(offset) if offset is not None else 0
        except (TypeError, ValueError):
            return 0
        return max(0, n)

    @staticmethod
    def _human_size(size: Any) -> str:
        try:
            n = int(size)
        except (TypeError, ValueError):
            return "未知"
        if n < 1024:
            return f"{n} B"
        if n < 1024 * 1024:
            return f"{n / 1024:.1f} KB"
        return f"{n / 1024 / 1024:.2f} MB"
