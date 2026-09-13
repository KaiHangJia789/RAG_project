"""
文档业务逻辑服务（Week5: DB + Cache 版）

改造要点:
  - Week4: self._documents: dict = {}  内存存储
  - Week5: DocumentRepository + DocumentCache + StorageService
  - 所有读写经过 Cache-Aside 模式
  - 写操作走事务（db.transaction），读操作缓存优先
"""
import asyncio
import logging
from datetime import datetime, UTC

from app.cache.document_cache import CacheAction, DocumentCache
from app.config import settings
from app.db.connection import Database
from app.db.repositories.document_repo import DocumentRepository
from app.exceptions.handlers import DocumentNotFoundError, UserNotFoundError
from app.models.document import (
    DocumentInfo,
    DocumentResponse,
    DocumentStatus,
)
from app.models.requests import DocumentSearchParams
from app.models.response import PaginatedData
from app.services.storage_service import StorageService

logger = logging.getLogger("rag_api.service")


class DocumentService:
    """
    文档服务 — DB + Cache 实现。

    读取流程（Cache-Aside）:
      Cache.get() → HIT → 返回
                   → DEFER → try_acquire_lock() → 成功 → 查 DB → set_and_release
                                                  → 失败 → wait_and_retry()
    写入流程:
      db.transaction() → repo.insert/update/delete → cache.invalidate()
    """

    def __init__(
        self,
        db: Database,
        cache: DocumentCache,
        storage: StorageService,
        doc_repo: DocumentRepository | None = None,
        default_username: str | None = None,
    ) -> None:
        self.db = db
        self.cache = cache
        self.storage = storage
        self.doc_repo = doc_repo or DocumentRepository()
        self._default_username = default_username or settings.DEFAULT_USERNAME
        # 解析结果缓存（user_id 在库的整个生命周期内不变）
        self._default_user_id: str | None = None
        self._user_id_lock = asyncio.Lock()

    # ═══════════════════════════════════════════════════════════
    # 默认用户解析
    # ═══════════════════════════════════════════════════════════

    async def _resolve_default_user_id(self) -> str:
        """
        按用户名查 users 表拿 user_id（带进程内缓存）。

        为什么不把 UUID 写成常量：UUID 由 seed.sql 用 gen_random_uuid() 生成，
        换库/重建库后会变成另一个值，硬编码必然失效并撞上 documents 的外键约束。
        按 username 查是环境无关的。

        Raises:
            UserNotFoundError: users 表里没有这个用户名
        """
        if self._default_user_id is not None:
            return self._default_user_id

        # 双检锁：并发请求只让一个去查库，其余等结果
        async with self._user_id_lock:
            if self._default_user_id is not None:
                return self._default_user_id

            row = await self.db.fetch_one(  # type: ignore[union-attr]
                "SELECT id FROM users WHERE username = $1", self._default_username
            )
            if row is None:
                # 不缓存失败结果：补跑 seed.sql 后无需重启即可自愈
                raise UserNotFoundError(self._default_username)

            self._default_user_id = str(row["id"])
            logger.info(
                "默认用户已解析: %s → %s", self._default_username, self._default_user_id
            )
            return self._default_user_id

    async def _user_id_or_default(self, user_id: str | None) -> str:
        """显式传入则用传入值，否则解析默认用户"""
        return user_id if user_id is not None else await self._resolve_default_user_id()

    # ═══════════════════════════════════════════════════════════
    # CREATE
    # ═══════════════════════════════════════════════════════════

    async def create_document(
        self,
        filename: str,
        content: bytes,
        user_id: str | None = None,
    ) -> DocumentResponse:
        """
        上传并创建文档（事务保证）。

        顺序很重要：校验 → 解析 user_id → 写盘 → 入库。
        纯计算的文件校验放最前（不浪费 DB 往返），写盘放在入库前一步，
        任何前置失败都不会留下孤儿文件。
        """
        # 1. 文件校验（纯计算，先拦掉非法请求）
        self.storage.validate_file(filename, len(content))

        # 2. 解析 user_id（users 表缺记录时在这里失败，此时还没写盘）
        owner_id = await self._user_id_or_default(user_id)

        # 3. 物理存储
        storage_result = await self.storage.save(filename, content)

        # 4. 写入数据库（事务）；失败则回滚并删掉刚落盘的文件，避免孤儿文件堆积
        try:
            async with self.db.transaction() as conn:
                record = await self.doc_repo.insert(
                    conn,
                    {
                        "user_id": owner_id,
                        "filename": filename,
                        "file_type": storage_result["file_type"],
                        "file_size": storage_result["file_size"],
                        "storage_path": storage_result["storage_path"],
                        "status": "uploaded",
                        "created_at": datetime.now(UTC),
                        "updated_at": datetime.now(UTC),
                    },
                )
        except Exception:
            try:
                self.storage.delete(storage_result["storage_path"])
            except OSError as cleanup_err:
                logger.warning(
                    "孤儿文件清理失败（需人工处理）: %s — %s",
                    storage_result["storage_path"], cleanup_err,
                )
            raise

        return self._record_to_response(record)

    # ═══════════════════════════════════════════════════════════
    # UPDATE
    # ═══════════════════════════════════════════════════════════

    async def update_status(self, doc_id: str, status: DocumentStatus) -> None:
        """
        更新文档处理状态（落库 + 失效缓存）。

        解析完成后必须调用，否则 DB 里状态永远停在 uploaded，
        列表页/详情页看不到真实的解析结果。
        """
        async with self.db.transaction() as conn:
            await self.doc_repo.update(
                conn,
                doc_id,
                {"status": status.value, "updated_at": datetime.now(UTC)},
            )
        await self.cache.invalidate(doc_id)

    # ═══════════════════════════════════════════════════════════
    # READ（缓存优先）
    # ═══════════════════════════════════════════════════════════

    async def get_document(self, doc_id: str) -> DocumentResponse | None:
        """
        获取文档详情（Cache-Aside）。

        流程:
          缓存命中 → 返回
          缓存未命中 → 抢锁 → 查DB回写 → 返回
                      没抢到锁 → 退避等待 → 返回
        """
        # Step 1: 读缓存
        action, data = await self.cache.get(doc_id)
        if action == CacheAction.HIT:
            if data is None:   # 空值缓存 → 文档确实不存在
                return None
            await self.cache.record_access(doc_id)
            return self._dict_to_response(data)

        # Step 2: 抢锁
        got_lock = await self.cache.try_acquire_lock(doc_id)
        if got_lock:
            # === 我是唯一执行者 → 查 DB 回写 ===
            try:
                async with self.db.pool.acquire() as conn:  # type: ignore[union-attr]
                    row = await self.doc_repo.get_by_id(conn, doc_id)
                if row:
                    doc_dict = dict(row)
                    await self.cache.set_and_release(doc_id, doc_dict)
                    await self.cache.record_access(doc_id)
                    return self._record_to_response(row)
                else:
                    await self.cache.set_null(doc_id)
                    return None
            except Exception:
                await self.cache.release_lock(doc_id)
                raise

        # Step 3: 等别人回写
        data = await self.cache.wait_and_retry(doc_id)
        if data:
            await self.cache.record_access(doc_id)
            return self._dict_to_response(data)

        # Step 4: 退避耗尽 → 返回 None（API 层返回 404/503）
        logger.warning("get_document 缓存等待耗尽: doc_id=%s", doc_id)
        return None

    # ═══════════════════════════════════════════════════════════
    # LIST（直接查 DB 分页，不做缓存 — 列表数据变化频繁）
    # ═══════════════════════════════════════════════════════════

    async def search_documents(
        self,
        params: DocumentSearchParams,
        user_id: str | None = None,
    ) -> PaginatedData[DocumentResponse]:
        """
        文档列表查询（Keyset 游标分页）。

        列表查询不做缓存，原因：
          - 过滤条件组合多变，缓存命中率极低
          - 数据更新频繁
        """
        owner_id = await self._user_id_or_default(user_id)

        async with self.db.pool.acquire() as conn:  # type: ignore[union-attr]
            rows, has_more, next_cursor = await self.doc_repo.search_by_user(
                conn,
                user_id=owner_id,
                keyword=params.keyword,
                status=params.status,
                file_type=params.file_type,
                cursor=None,  # 首頁
                page_size=params.page_size,
            )

        items = [self._record_to_response(r) for r in rows]
        total = await self.db.fetch_one(  # type: ignore[union-attr]
            "SELECT COUNT(*) FROM documents WHERE user_id = $1", owner_id
        )
        total_count = total[0] if total else 0

        return PaginatedData(
            items=items,
            total=total_count,
            page=params.page,
            page_size=params.page_size,
            total_pages=max(1, (total_count + params.page_size - 1) // params.page_size),
        )

    # ═══════════════════════════════════════════════════════════
    # DELETE
    # ═══════════════════════════════════════════════════════════

    async def delete_document(self, doc_id: str) -> bool:
        """
        删除文档记录 + 物理文件 + 缓存。

        顺序：先在事务里读出 storage_path 并删记录，提交后再删磁盘文件。
        必须先删 DB 再删文件 —— 反过来的话事务一旦回滚，就会留下
        "记录还在、文件没了" 的坏数据（详情页能查到但文件读不出来）。
        chunks 表有 ON DELETE CASCADE，会自动跟着删。
        """
        async with self.db.transaction() as conn:
            row = await self.doc_repo.get_by_id(conn, doc_id)
            if row is None:
                return False
            storage_path = dict(row)["storage_path"]
            await self.doc_repo.delete(conn, doc_id)

        # 事务已提交 —— 再删物理文件（漏这步就会不断堆积孤儿文件）
        try:
            self.storage.delete(storage_path)
        except OSError as e:
            # 记录已删，删除语义已达成；文件残留需告警人工处理
            logger.warning(
                "物理文件删除失败（遗留孤儿文件）: %s — %s", storage_path, e
            )

        # 缓存失效（事务外，失败不影响 DB 一致性）
        await self.cache.invalidate(doc_id)
        return True

    # ═══════════════════════════════════════════════════════════
    # INFO（轻量版）
    # ═══════════════════════════════════════════════════════════

    async def get_document_info(self, doc_id: str) -> DocumentInfo | None:
        """获取文档元信息（轻量版，不含存储路径）"""
        doc = await self.get_document(doc_id)
        if doc is None:
            return None
        return DocumentInfo(
            id=doc.id,
            filename=doc.filename,
            file_type=doc.file_type,
            file_size=doc.file_size,
            status=doc.status,
            created_at=doc.created_at,
            updated_at=doc.updated_at,
        )

    # ═══════════════════════════════════════════════════════════
    # 辅助转换方法
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _record_to_response(record) -> DocumentResponse:
        """asyncpg Record → DocumentResponse"""
        d = dict(record)
        return DocumentResponse(
            id=str(d["id"]),
            filename=d["filename"],
            file_type=d["file_type"],
            file_size=d["file_size"],
            status=d["status"],
            storage_path=d["storage_path"],
            created_at=d["created_at"],
            updated_at=d["updated_at"],
        )

    @staticmethod
    def _dict_to_response(data: dict[str, str]) -> DocumentResponse:
        """Redis 返回的 dict（值都是字符串） → DocumentResponse"""
        return DocumentResponse(
            id=data["id"],
            filename=data["filename"],
            file_type=data["file_type"],
            file_size=int(data["file_size"]),
            status=DocumentStatus(data["status"]),
            storage_path=data["storage_path"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )
