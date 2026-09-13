"""
Embedding 客户端抽象基类
"""
from abc import ABC, abstractmethod


class EmbeddingClient(ABC):
    """向量化客户端抽象接口"""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        批量向量化。

        Args:
            texts: 待向量化的文本列表

        Returns:
            与输入一一对应的向量列表（每个是 dim 维 float 列表）
        """
        ...
