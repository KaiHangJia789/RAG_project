"""
数据库连接池管理器
封装 asyncpg 连接池的初始化、事务管理和资源释放
"""
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncpg

logger = logging.getLogger("rag_api.db")


class Database:
    """
    PostgreSQL 连接池管理器。

    Usage:
        db = Database()
        await db.connect(settings.DATABASE_DSN)

        # 单查询
        row = await db.fetch_one("SELECT * FROM documents WHERE id = $1", doc_id)

        # 事务（跨表原子操作）
        async with db.transaction() as conn:
            await db.execute(conn, "INSERT INTO documents ...", ...)
            await db.execute(conn, "INSERT INTO chunks ...", ...)
            # 任何一步失败 → 全部回滚

        await db.disconnect()
    """

    def __init__(self) -> None:
        self.pool: asyncpg.Pool | None = None

    async def connect(self, dsn: dict) -> None:
        """初始化连接池"""
        self.pool = await asyncpg.create_pool(
            **dsn,
            min_size=2,
            max_size=10,
            command_timeout=30,
            ssl=False,  # 云服务器 SSL=off，关闭 SSL 避免 WinError 121 超时
        )
        logger.info("PostgreSQL 连接池已建立 (min=2, max=10)")

    async def disconnect(self) -> None:
        """关闭连接池"""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL 连接池已关闭")

    # ── 单查询方法（自动 acquire/release）────────────────────

    async def fetch_one(self, query: str, *args) -> asyncpg.Record | None:
        """执行查询并返回单行"""
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetch_all(self, query: str, *args) -> list[asyncpg.Record]:
        """执行查询并返回全部行"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return list(rows)

    async def execute(self, query: str, *args) -> str:
        """执行写操作，返回状态字符串"""
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    # ── 事务上下文管理器 ────────────────────────────────────

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[asyncpg.Connection]:
        """
        事务上下文管理器。

        在同一个连接和事务中执行多个操作。
        with 块正常退出时自动 COMMIT，抛异常时自动 ROLLBACK。

        Usage:
            async with db.transaction() as conn:
                await doc_repo.insert(conn, data)
                await chunk_repo.insert(conn, chunk_data)
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                yield conn
