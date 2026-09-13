"""
阿里云百炼 Embedding 客户端（qwen3.7-text-embedding-flash）

OpenAI 兼容接口，复用 openai SDK + langfuse.openai 包装器做全链路追踪。

安全措施:
  - API Key 从 settings 读取，不硬编码
  - 批量限流：分批调用，避免单次超 API 输入上限
  - 网络重试：瞬时抖动自动重试
  - 维度校验：返回维度与配置不一致时抛异常，防止 FAISS 索引错位
"""
import asyncio
import logging

from app.config import settings
from app.embedding.base import EmbeddingClient
from app.llm.observability import langfuse_enabled

logger = logging.getLogger("rag_api.embedding")


class DashscopeEmbeddingClient(EmbeddingClient):
    """阿里云百炼 Embedding 客户端"""

    def __init__(self, batch_size: int = 50, max_retries: int = 3):
        """
        Args:
            batch_size: 单次 API 调用的最大文本条数（限流）
            max_retries: 网络失败最大重试次数
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size 必须 > 0，实际 {batch_size}")
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._model = settings.EMBEDDING_MODEL
        self._dim = settings.EMBEDDING_DIM

        # LangFuse 追踪：有 key 用包装器（自动记录 token/耗时），无 key 降级
        if langfuse_enabled():
            from langfuse.openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=settings.DASHSCOPE_API_KEY,
                base_url=settings.EMBEDDING_BASE_URL,
            )
        else:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=settings.DASHSCOPE_API_KEY,
                base_url=settings.EMBEDDING_BASE_URL,
            )

    # ── 主接口 ──

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        批量向量化（自动分批 + 重试 + 维度校验）。

        Args:
            texts: 待向量化文本列表（可为空）

        Returns:
            向量列表，与输入一一对应
        """
        if not texts:
            return []

        results: list[list[float]] = []
        # 分批处理，控制单次调用规模
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_vecs = await self._embed_batch(batch)
            results.extend(batch_vecs)

        return results

    async def embed_one(self, text: str) -> list[float]:
        """单条向量化"""
        return (await self.embed([text]))[0]

    # ── 内部实现 ──

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """单批向量化（带重试；维度校验错误直接冒泡，不重试）"""
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return await self._call_api(texts)
            except ValueError:
                # 维度校验失败是配置错误，重试无意义，直接冒泡
                raise
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    logger.warning(
                        "embedding 调用失败（第 %d/%d 次）: %s，1 秒后重试",
                        attempt, self.max_retries, e,
                    )
                    await asyncio.sleep(1)
        raise RuntimeError(f"embedding 调用最终失败: {last_err}")

    async def _call_api(self, texts: list[str]) -> list[list[float]]:
        """单次 API 调用 + 维度校验"""
        resp = await self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dim,
        )
        # 按 index 排序（API 返回顺序可能乱序）
        data = sorted(resp.data, key=lambda d: d.index)
        vectors = [d.embedding for d in data]

        # 维度校验（安全措施：防止 FAISS 索引维度错位）
        for i, vec in enumerate(vectors):
            if len(vec) != self._dim:
                raise ValueError(
                    f"embedding 维度不匹配：期望 {self._dim}，实际 {len(vec)}（第 {i} 条）"
                )
        return vectors
