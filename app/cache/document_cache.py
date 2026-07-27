"""
文档缓存层 — Cache-Aside 模式

核心职责:
  - 读: HIT → 返回  /  DEFER → 上层等待或抢锁
  - 锁: try_acquire_lock 抢锁 → 查 DB → set_and_release 回写
  - 等待: wait_and_retry 指数退避轮询
  - 失效: invalidate 统一删除所有相关缓存

防护机制:
  - TTL 随机抖动（防雪崩）
  - 互斥锁（防击穿）
  - 空值缓存（防穿透）
  - 退避不降级（防降级风暴）
  - finally 释锁（防死锁）
"""
import asyncio
import logging
import random
from enum import Enum
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger("rag_api.cache.doc")


class CacheAction(Enum):
    """缓存读取结果的动作指示"""
    HIT = "hit"       # 缓存命中，数据可用
    DEFER = "defer"   # 缓存未命中，需要抢锁或等待


class DocumentCache:
    """文档缓存操作封装"""

    # ── TTL 配置 ──────────────────────────────────────────
    BASE_TTL = 600          # 基础 TTL: 10 分钟
    TTL_JITTER = 60         # 随机抖动: ±60 秒 → 实际 540s ~ 660s
    NULL_TTL = 60           # 空值缓存: 60 秒（防穿透）
    HOT_TOP_N = 20          # 热门文档 TOP N

    # ── 互斥锁配置 ────────────────────────────────────────
    LOCK_TTL = 5            # 锁超时: 5 秒
    POLL_INTERVAL = 0.02    # 轮询间隔: 20ms
    MAX_RETRIES = 25        # 最大重试次数
    #   retry 1-15: 固定 20ms 间隔（合计 300ms 快速阶段）
    #   retry 16-25: 指数退避 40ms→200ms（合计 ~800ms 慢速阶段）
    #   总等待时间上限: ~1.1s

    def __init__(self, redis: aioredis.Redis) -> None:
        self.redis = redis

    # ═══════════════════════════════════════════════════════════
    # Key 命名规范
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _meta_key(doc_id: str) -> str:
        return f"doc:meta:{doc_id}"

    @staticmethod
    def _lock_key(doc_id: str) -> str:
        return f"lock:doc:{doc_id}"

    @staticmethod
    def _hot_key() -> str:
        return "doc:hot"

    # ═══════════════════════════════════════════════════════════
    # 读缓存（不参与锁逻辑）
    # ═══════════════════════════════════════════════════════════

    async def get(self, doc_id: str) -> tuple[CacheAction, dict[str, str] | None]:
        """
        纯读缓存，返回 (动作, 数据)。

        - HIT  + dict → 直接用
        - DEFER + None → 没命中，上层决定抢锁还是等待
        """
        key = self._meta_key(doc_id)
        data = await self.redis.hgetall(key)
        if data:
            # 检查是否为 NULL 占位（防穿透的空值缓存）
            if data.get("__null__") == "1":
                return CacheAction.HIT, None  # 空值命中，返回 None 表示"确认不存在"
            return CacheAction.HIT, data
        return CacheAction.DEFER, None

    # ═══════════════════════════════════════════════════════════
    # 互斥锁（抢锁 + 等待 + 释放）
    # ═══════════════════════════════════════════════════════════

    async def try_acquire_lock(self, doc_id: str) -> bool:
        """
        尝试获取"查 DB 回写"的排他权限。

        - True  → 你是唯一的执行者，去查 DB 然后回写缓存
        - False → 别人持有锁，调用 wait_and_retry 等待
        """
        lock_key = self._lock_key(doc_id)
        # SETNX: 原子操作，只有一个协程能成功
        return await self.redis.set(lock_key, "1", nx=True, ex=self.LOCK_TTL)

    async def wait_and_retry(
        self, doc_id: str, max_retries: int = MAX_RETRIES
    ) -> dict[str, str] | None:
        """
        轮询等待缓存被回写（带指数退避，不降级查 DB）。

        策略:
          retry 1-15:   固定 20ms 间隔（快速等待，合计 300ms）
          retry 16-25:  指数退避 40ms→200ms（合计 ~800ms）
          超过 max_retries: 返回 None（Service 层决定是否返回 503）

        关键：此方法绝不直接查 DB，该决定权完全交给 Service 层。
        """
        for attempt in range(1, max_retries + 1):
            # 先看一眼缓存
            key = self._meta_key(doc_id)
            data = await self.redis.hgetall(key)
            if data:
                if data.get("__null__") == "1":
                    return None
                return data

            # 动态退避
            if attempt <= 15:
                delay = self.POLL_INTERVAL                     # 20ms
            else:
                base = self.POLL_INTERVAL * (2 ** (attempt - 15))
                delay = min(base, 0.2)                         # 上限 200ms

            await asyncio.sleep(delay)

        logger.warning(
            "cache_wait_exhausted doc_id=%s retries=%d", doc_id, max_retries
        )
        return None

    async def release_lock(self, doc_id: str) -> None:
        """释放互斥锁（幂等：锁已删除时 DELETE 空操作）"""
        await self.redis.delete(self._lock_key(doc_id))

    # ═══════════════════════════════════════════════════════════
    # 回写缓存 + 释放锁
    # ═══════════════════════════════════════════════════════════

    def _ttl(self) -> int:
        """TTL 随机抖动: 540s ~ 660s"""
        return self.BASE_TTL + random.randint(-self.TTL_JITTER, self.TTL_JITTER)

    async def set_and_release(self, doc_id: str, data: dict[str, Any]) -> None:
        """
        回写缓存 + 释放锁（pipeline 原子执行）。

        写入时将 dict 的所有值转为字符串（redis-py 需要）。
        """
        key = self._meta_key(doc_id)
        lock_key = self._lock_key(doc_id)
        ttl = self._ttl()

        # 所有值转为字符串
        str_data = {k: str(v) for k, v in data.items()}

        async with self.redis.pipeline() as pipe:
            pipe.hset(key, mapping=str_data)
            pipe.expire(key, ttl)
            pipe.delete(lock_key)

    async def set_null(self, doc_id: str) -> None:
        """
        空值缓存：标记此 ID 在 DB 中确实不存在。
        防止不存在的 ID 被反复查询穿透到 DB。
        """
        key = self._meta_key(doc_id)
        lock_key = self._lock_key(doc_id)
        async with self.redis.pipeline() as pipe:
            pipe.hset(key, "__null__", "1")
            pipe.expire(key, self.NULL_TTL)
            pipe.delete(lock_key)

    # ═══════════════════════════════════════════════════════════
    # 缓存失效（所有写操作统一入口）
    # ═══════════════════════════════════════════════════════════

    async def invalidate(self, doc_id: str) -> None:
        """
        删除文档的所有相关缓存。

        调用时机: UPDATE / DELETE 操作后。
        """
        async with self.redis.pipeline() as pipe:
            pipe.delete(self._meta_key(doc_id))
            pipe.zrem(self._hot_key(), doc_id)

    # ═══════════════════════════════════════════════════════════
    # 热门文档排行（Sorted Set）
    # ═══════════════════════════════════════════════════════════

    async def record_access(self, doc_id: str) -> None:
        """记录一次文档访问（对热门排行计数 +1）"""
        await self.redis.zincrby(self._hot_key(), 1, doc_id)

    async def get_hot_documents(self, top_n: int = HOT_TOP_N) -> list[tuple[str, float]]:
        """获取热门文档 TOP N（返回 doc_id + score）"""
        return await self.redis.zrevrange(
            self._hot_key(), 0, top_n - 1, withscores=True
        )
