"""
Redis 连接管理器
封装 redis-py 异步连接
"""
import logging

import redis.asyncio as aioredis

logger = logging.getLogger("rag_api.cache")


class RedisClient:
    """Redis 异步客户端"""

    def __init__(self) -> None:
        self.client: aioredis.Redis | None = None

    async def connect(self, url: str) -> None:
        """建立 Redis 连接"""
        self.client = aioredis.from_url(
            url,
            encoding="utf-8",
            decode_responses=True,
        )
        # 验证连接
        await self.client.ping()
        logger.info("Redis 连接已建立: %s", url)

    async def disconnect(self) -> None:
        """关闭 Redis 连接"""
        if self.client:
            await self.client.aclose()
            logger.info("Redis 连接已关闭")
