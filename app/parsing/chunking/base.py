"""
Chunk 切分抽象基类

定义统一的数据结构和策略接口。
所有策略实现 chunk(text) -> list[Chunk]。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    """切分出的一个文本块"""
    text: str
    start: int                                  # 在原文中的起始偏移
    end: int                                    # 在原文中的结束偏移（不含）

    @property
    def length(self) -> int:
        return len(self.text)


class ChunkStrategy(ABC):
    """
    Chunk 策略抽象基类。

    子类必须实现：
      - name: 策略唯一标识
      - chunk(text): 核心切分方法
    """

    name: str = "base"

    @abstractmethod
    def chunk(self, text: str) -> list[Chunk]:
        """将文本切分为块列表"""
        ...

    # ── 通用边界校验（子类可复用） ──

    @staticmethod
    def _validate_text(text: str) -> None:
        """空文本校验（边界防护）"""
        if not text or not text.strip():
            return  # 空文本合法，切分结果为空列表

    @staticmethod
    def _validate_params(*, chunk_size: int, overlap: int) -> None:
        """
        切分参数校验（防死循环/防越界）。

        Raises:
            ValueError: chunk_size <= 0 或 overlap >= chunk_size
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size 必须 > 0，实际 {chunk_size}")
        if overlap < 0:
            raise ValueError(f"overlap 必须 >= 0，实际 {overlap}")
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap({overlap}) 必须 < chunk_size({chunk_size})，否则会死循环"
            )
