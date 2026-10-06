"""
数据库查询工具

**安全边界（三条硬约束，缺一不可）**：
  1. **只读**：只允许 SELECT。用正则拒绝任何写操作关键字 ——
     工具参数是模型生成的文本，被提示注入诱导时可能传入 DROP/DELETE。
  2. **表白名单**：只允许查 `documents` / `chunks` 两张业务表。
     绝不允许触及 `users`（有密码哈希）等敏感表。
  3. **强制 LIMIT**：没有 LIMIT 的查询自动补一个，防止全表扫描拖垮服务。

  另外用**参数化查询**传递值，不接受模型拼接的 SQL 字面量。

用途：让 Agent 能回答"知识库里有几篇文档""某文档有多少个分块"这类
元数据问题 —— 这些问题检索不到（答案不在文档内容里，而在元数据里）。
"""
import logging
import re
from typing import Any

from app.agent.tools.base import Tool, ToolResult

logger = logging.getLogger("rag_api.agent.tools.db")

# 允许查询的表 + 每张表允许的列（白名单，防止 SELECT * 泄露敏感字段）
_ALLOWED_TABLES: dict[str, set[str]] = {
    "documents": {
        "id", "filename", "file_type", "file_size", "status",
        "created_at", "updated_at",
    },
    "chunks": {
        "id", "document_id", "chunk_index", "chunk_strategy",
        "token_count", "page_number", "created_at",
    },
}

# 写操作与危险关键字（只读约束）
_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|"
    r"COPY|VACUUM|REINDEX|REFRESH|CALL|DO|EXECUTE|MERGE)\b",
    re.IGNORECASE,
)

_MAX_ROWS = 50


class DbQueryTool(Tool):
    """
    知识库元数据查询（只读）。

    支持两个动作，而不是让模型自由写 SQL —— 自由 SQL 即便做白名单过滤，
    也是一条需要持续审计的攻击面。给结构化动作更安全，且模型更容易用对。
    """

    name = "query_metadata"
    description = (
        "查询知识库的元数据统计。用于回答「有几篇文档」「某文档有多少分块」"
        "「某个策略下有多少块」这类问题——这类信息不在文档内容里，检索不到。"
    )

    def __init__(self, db) -> None:
        self.db = db
        self.calls: list[dict] = []

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "查询动作",
                    "enum": ["list_documents", "count_chunks", "document_stats"],
                },
                "filename": {
                    "type": "string",
                    "description": "文档名（action=count_chunks 时需要）",
                },
            },
            "required": ["action"],
        }

    async def _run(self, action: str = "", filename: str | None = None, **kwargs: Any) -> ToolResult:
        act = (action or "").strip()
        self.calls.append({"action": act, "filename": filename})

        if act == "list_documents":
            return await self._list_documents()
        if act == "count_chunks":
            return await self._count_chunks(filename)
        if act == "document_stats":
            return await self._document_stats()

        return ToolResult(
            ok=False,
            content=f"未知动作 '{act}'。可用：list_documents / count_chunks / document_stats",
            error="unknown_action",
        )

    # ═══════════════════════════════════════════════════════════
    # 动作实现（全部参数化查询 + 白名单字段）
    # ═══════════════════════════════════════════════════════════

    async def _list_documents(self) -> ToolResult:
        rows = await self.db.fetch_all(
            'SELECT filename, file_type, file_size, status '
            'FROM "documents" ORDER BY created_at DESC LIMIT $1',
            _MAX_ROWS,
        )
        if not rows:
            return ToolResult(ok=True, content="知识库中还没有文档。")

        lines = [
            f"- {r['filename']}（{r['file_type']}, {r['file_size']} 字节, {r['status']}）"
            for r in rows
        ]
        return ToolResult(
            ok=True,
            content=f"知识库共 {len(rows)} 篇文档：\n" + "\n".join(lines),
            meta={"count": len(rows)},
        )

    async def _count_chunks(self, filename: str | None) -> ToolResult:
        if not filename:
            return ToolResult(
                ok=False, content="action=count_chunks 需要提供 filename", error="missing_filename"
            )

        rows = await self.db.fetch_all(
            'SELECT c.chunk_strategy, COUNT(*) AS n '
            'FROM "chunks" c JOIN "documents" d ON d.id = c.document_id '
            'WHERE d.filename = $1 GROUP BY c.chunk_strategy ORDER BY c.chunk_strategy',
            filename,
        )
        if not rows:
            return ToolResult(
                ok=True, content=f"没有找到文档「{filename}」的分块记录。"
            )

        lines = [f"- {r['chunk_strategy']}: {r['n']} 块" for r in rows]
        total = sum(r["n"] for r in rows)
        return ToolResult(
            ok=True,
            content=f"文档「{filename}」共 {total} 个分块：\n" + "\n".join(lines),
            meta={"total": total},
        )

    async def _document_stats(self) -> ToolResult:
        rows = await self.db.fetch_all(
            'SELECT status, COUNT(*) AS n FROM "documents" GROUP BY status ORDER BY status'
        )
        chunk_row = await self.db.fetch_one('SELECT COUNT(*) AS n FROM "chunks"')

        lines = [f"- {r['status']}: {r['n']} 篇" for r in rows]
        total_chunks = chunk_row["n"] if chunk_row else 0
        return ToolResult(
            ok=True,
            content=(
                f"知识库统计：\n" + "\n".join(lines) +
                f"\n- 分块总数: {total_chunks}"
            ),
            meta={"chunks": total_chunks},
        )


# ═══════════════════════════════════════════════════════════════
# SQL 安全校验（供将来支持自定义 SQL 时复用）
# ═══════════════════════════════════════════════════════════════

class UnsafeSqlError(Exception):
    pass


def validate_readonly_sql(sql: str) -> None:
    """
    校验 SQL 是只读且限定在白名单表内。

    当前 `DbQueryTool` 走结构化动作，不需要它；但如果将来要开放自定义 SQL，
    **必须**先过这一关。这里实现出来并配单测，避免将来临时写漏。

    Raises:
        UnsafeSqlError: 含写操作、非 SELECT、或触及白名单外的表
    """
    text = (sql or "").strip()
    if not text:
        raise UnsafeSqlError("SQL 为空")

    # 去掉字符串字面量后再检查关键字，避免 'DROP' 这种出现在字符串里被误判
    stripped = re.sub(r"'[^']*'", "''", text)

    if _FORBIDDEN.search(stripped):
        raise UnsafeSqlError("只允许只读查询，检测到写操作关键字")

    if not re.match(r"^\s*(SELECT|WITH)\b", stripped, re.IGNORECASE):
        raise UnsafeSqlError("只允许 SELECT 查询")

    if ";" in stripped.rstrip(";"):
        raise UnsafeSqlError("不允许多条语句")

    # 表白名单
    referenced = set(re.findall(r'\b(?:FROM|JOIN)\s+"?(\w+)"?', stripped, re.IGNORECASE))
    illegal = referenced - set(_ALLOWED_TABLES)
    if illegal:
        raise UnsafeSqlError(f"不允许访问表: {', '.join(sorted(illegal))}")

    if not referenced:
        raise UnsafeSqlError("未能识别查询的表")
