"""
ChunkRepository — chunk 数据访问层

继承 BaseRepository 获得通用 CRUD；本类补充切分/索引/检索所需的查询。

注意（写新方法时务必遵守）：
  - **只用单行 VALUES**。测试替身 FakeConnection 用
    `re.search(r'\\(([^)]+)\\)\\s*VALUES', sql)` 抓列名，多行 VALUES 会匹配错。
  - INSERT 的批量场景用循环调用 `insert()`，不要在一条 SQL 里塞多组值。
"""
import asyncpg

from app.db.repositories.base import BaseRepository


class ChunkRepository(BaseRepository):
    table_name = "chunks"

    # ── 写入 ──────────────────────────────────────────────

    async def insert_chunk(
        self, conn: asyncpg.Connection, data: dict
    ) -> asyncpg.Record:
        """
        插入单个 chunk，返回完整行。

        `faiss_id` 走表默认值 `nextval('chunks_faiss_id_seq')`，**不要**在 data 里传，
        由 PG 保证全局唯一（应用层计数器重启后会重号，而 FAISS 对重复 id 不报错，
        只会静默产生两条共享 id 的向量）。

        调用方取 id 时用 `row.get("faiss_id")` —— 测试替身只把 SQL 列清单里的列
        放进返回行，用下标访问会 KeyError。
        """
        return await self.insert(conn, data)

    # ── 查询 ──────────────────────────────────────────────

    async def fetch_for_indexing(
        self, conn: asyncpg.Connection, strategy: str
    ) -> list[asyncpg.Record]:
        """
        取某策略下所有已分配 faiss_id 的 chunk + 所属文档名（建索引用）。

        JOIN 只出现在脚本/重建路径（连真库），不在在线检索路径上 ——
        测试替身对 JOIN 支持有限，把 JOIN 挡在在线路径外可保证可测性。
        """
        return list(await conn.fetch(
            """
            SELECT c.id, c.faiss_id, c.document_id, c.chunk_index, c.chunk_text,
                   c.page_number, c.chunk_strategy, d.filename
            FROM "chunks" c
            JOIN "documents" d ON d.id = c.document_id
            WHERE c.chunk_strategy = $1 AND c.faiss_id IS NOT NULL
            ORDER BY c.faiss_id
            """,
            strategy,
        ))

    async def fetch_by_faiss_ids(
        self, conn: asyncpg.Connection, faiss_ids: list[int]
    ) -> list[asyncpg.Record]:
        """按 FAISS id 批量取 chunk（检索命中后的回表）"""
        if not faiss_ids:
            return []
        return list(await conn.fetch(
            'SELECT * FROM "chunks" WHERE faiss_id = ANY($1::bigint[])',
            faiss_ids,
        ))

    async def fetch_by_document(
        self,
        conn: asyncpg.Connection,
        document_id: str,
        strategy: str | None = None,
    ) -> list[asyncpg.Record]:
        """取某文档的全部 chunk（按 chunk_index 升序）"""
        if strategy is None:
            return list(await conn.fetch(
                'SELECT * FROM "chunks" WHERE document_id = $1 ORDER BY chunk_index',
                document_id,
            ))
        return list(await conn.fetch(
            'SELECT * FROM "chunks" '
            'WHERE document_id = $1 AND chunk_strategy = $2 ORDER BY chunk_index',
            document_id,
            strategy,
        ))

    async def fetch_neighbors(
        self,
        conn: asyncpg.Connection,
        document_id: str,
        chunk_index: int,
        *,
        before: int = 1,
        after: int = 1,
    ) -> list[asyncpg.Record]:
        """取目标 chunk 及其前后邻居（引用可定位：展开上下文用）"""
        return list(await conn.fetch(
            'SELECT * FROM "chunks" '
            'WHERE document_id = $1 AND chunk_index BETWEEN $2 AND $3 '
            'ORDER BY chunk_index',
            document_id,
            max(0, chunk_index - before),
            chunk_index + after,
        ))

    async def fetch_one_by_id(
        self, conn: asyncpg.Connection, chunk_id: str
    ) -> asyncpg.Record | None:
        """按 chunk 主键取单条（含文件名）"""
        return await conn.fetchrow(
            """
            SELECT c.*, d.filename
            FROM "chunks" c
            JOIN "documents" d ON d.id = c.document_id
            WHERE c.id = $1
            """,
            chunk_id,
        )

    # ── 统计 ──────────────────────────────────────────────

    async def count_indexed(
        self, conn: asyncpg.Connection, strategy: str
    ) -> int:
        """统计某策略下已分配 faiss_id 的 chunk 数（索引一致性自检用）"""
        row = await conn.fetchrow(
            'SELECT COUNT(*) FROM "chunks" '
            'WHERE chunk_strategy = $1 AND faiss_id IS NOT NULL',
            strategy,
        )
        return row[0] if row else 0

    async def hash_exists(
        self, conn: asyncpg.Connection, document_id: str, chunk_hash: str
    ) -> bool:
        """判断同一文档下该文本是否已入库（避免重复向量化）"""
        row = await conn.fetchrow(
            'SELECT 1 FROM "chunks" WHERE document_id = $1 AND chunk_hash = $2 LIMIT 1',
            document_id,
            chunk_hash,
        )
        return row is not None
