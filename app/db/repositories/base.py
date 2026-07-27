"""
BaseRepository — 通用数据访问基类
子类必须定义 table_name 类属性
"""
from typing import Any

import asyncpg


class BaseRepository:
    """
    通用 Repository 基类。

    子类用法：
        class DocumentRepository(BaseRepository):
            table_name = "documents"

    conn 由上层（Service）通过事务上下文注入，Repository 不负责获取连接。
    """

    # 子类必须覆盖
    table_name: str

    @staticmethod
    def _quote(name: str) -> str:
        """安全转义 PostgreSQL 标识符（双引号包裹）。

        注意：table_name 和 column_name 始终是代码中的硬编码字面量，
        不是用户输入，不存在 SQL 注入风险。双引号包裹是为了：
        1. 保留大小写（PostgreSQL 默认将未引用标识符转为小写）
        2. 避免与 PostgreSQL 保留字冲突
        """
        return f'"{name}"'

    # ── CRUD ───────────────────────────────────────────────

    async def get_by_id(
        self, conn: asyncpg.Connection, id: str
    ) -> asyncpg.Record | None:
        """按主键查询"""
        tbl = self._quote(self.table_name)
        return await conn.fetchrow(f"SELECT * FROM {tbl} WHERE id = $1", id)

    async def insert(
        self, conn: asyncpg.Connection, data: dict[str, Any]
    ) -> asyncpg.Record:
        """
        插入记录并返回完整行。

        Args:
            data: 列名 → 值的映射
        Returns:
            插入后的完整记录
        """
        tbl = self._quote(self.table_name)
        columns = list(data.keys())
        values = list(data.values())
        placeholders = ", ".join(f"${i}" for i in range(1, len(columns) + 1))
        cols_quoted = ", ".join(
            self._quote(col) for col in columns
        )

        return await conn.fetchrow(
            f"INSERT INTO {tbl} ({cols_quoted}) VALUES ({placeholders}) RETURNING *",
            *values,
        )

    async def update(
        self, conn: asyncpg.Connection, id: str, data: dict[str, Any]
    ) -> asyncpg.Record | None:
        """更新记录并返回更新后的行"""
        tbl = self._quote(self.table_name)
        set_clause = ", ".join(
            f"{self._quote(col)} = ${i}"
            for i, col in enumerate(data.keys(), start=1)
        )
        values = list(data.values()) + [id]

        return await conn.fetchrow(
            f"UPDATE {tbl} SET {set_clause} WHERE id = ${len(values)} RETURNING *",
            *values,
        )

    async def delete(self, conn: asyncpg.Connection, id: str) -> bool:
        """删除记录，返回是否成功"""
        tbl = self._quote(self.table_name)
        result = await conn.fetchrow(
            f"DELETE FROM {tbl} WHERE id = $1 RETURNING id", id
        )
        return result is not None

    async def count(self, conn: asyncpg.Connection, where: str = "", *args) -> int:
        """统计记录数"""
        tbl = self._quote(self.table_name)
        clause = f"WHERE {where}" if where else ""
        row = await conn.fetchrow(
            f"SELECT COUNT(*) FROM {tbl} {clause}", *args
        )
        return row[0] if row else 0
