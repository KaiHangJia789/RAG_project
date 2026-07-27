"""
DocumentRepository — 文档数据访问层
扩展 BaseRepository，添加文档特有的查询方法（搜索/Keyset分页）
"""
import base64
import json
from datetime import datetime

import asyncpg

from app.db.repositories.base import BaseRepository


class DocumentRepository(BaseRepository):
    table_name = "documents"

    # ── Keyset 分页查询 ────────────────────────────────────

    async def search_by_user(
        self,
        conn: asyncpg.Connection,
        user_id: str,
        *,
        keyword: str | None = None,
        status: str | None = None,
        file_type: str | None = None,
        cursor: str | None = None,
        page_size: int = 20,
    ) -> tuple[list[asyncpg.Record], bool, str | None]:
        """
        按用户分页查询文档列表（Keyset 游标分页）。

        永远只扫描 page_size+1 行，不随 OFFSET 增大而性能下降。

        Args:
            cursor: base64 编码的游标（上一页最后一条记录的排序键）
            page_size: 每页数量

        Returns:
            (items, has_more, next_cursor)
        """
        tbl = self._quote("documents")
        conditions = [f"{tbl}.user_id = $1"]
        params: list = [user_id]
        param_idx = 2

        # 关键字搜索
        if keyword:
            conditions.append(f"{tbl}.filename ILIKE ${param_idx}")
            params.append(f"%{keyword}%")
            param_idx += 1

        # 状态过滤
        if status:
            conditions.append(f"{tbl}.status = ${param_idx}::document_status")
            params.append(status)
            param_idx += 1

        # 类型过滤
        if file_type:
            conditions.append(f"{tbl}.file_type = ${param_idx}")
            params.append(file_type)
            param_idx += 1

        # Keyset 游标解码
        if cursor:
            cursor_data = self._decode_cursor(cursor)
            conditions.append(
                f"({tbl}.created_at, {tbl}.id) < "
                f"(${param_idx}::timestamptz, ${param_idx + 1}::uuid)"
            )
            params.extend([cursor_data["created_at"], cursor_data["id"]])
            param_idx += 2

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT * FROM {tbl}
            WHERE {where_clause}
            ORDER BY {tbl}.created_at DESC, {tbl}.id DESC
            LIMIT ${param_idx}
        """
        params.append(page_size + 1)  # 多取 1 条判断 has_more

        rows = await conn.fetch(query, *params)

        has_more = len(rows) > page_size
        items = rows[:page_size]

        # 编码下一页游标
        next_cursor = None
        if has_more and items:
            next_cursor = self._encode_cursor(
                created_at=items[-1]["created_at"],
                doc_id=items[-1]["id"],
            )

        return items, has_more, next_cursor

    # ── 按 ID 批量查询 ─────────────────────────────────────

    async def get_by_ids(
        self, conn: asyncpg.Connection, ids: list[str]
    ) -> list[asyncpg.Record]:
        """批量查询文档（用于缓存回填）"""
        tbl = self._quote(self.table_name)
        rows = await conn.fetch(
            f"SELECT * FROM {tbl} WHERE id = ANY($1::uuid[])", ids
        )
        return list(rows)

    # ── 按状态统计 ─────────────────────────────────────────

    async def count_by_status(
        self, conn: asyncpg.Connection, user_id: str
    ) -> dict[str, int]:
        """统计各状态的文档数量"""
        tbl = self._quote(self.table_name)
        rows = await conn.fetch(
            f"""
            SELECT status, COUNT(*) as cnt
            FROM {tbl}
            WHERE user_id = $1
            GROUP BY status
            """,
            user_id,
        )
        return {row["status"]: row["cnt"] for row in rows}

    # ── 游标编解码辅助 ─────────────────────────────────────

    @staticmethod
    def _encode_cursor(created_at: datetime, doc_id: str) -> str:
        """将排序键编码为 base64 游标"""
        payload = json.dumps({
            "created_at": created_at.isoformat(),
            "id": str(doc_id),
        })
        return base64.urlsafe_b64encode(payload.encode()).decode()

    @staticmethod
    def _decode_cursor(cursor: str) -> dict:
        """解码 base64 游标"""
        payload = base64.urlsafe_b64decode(cursor.encode()).decode()
        return json.loads(payload)
