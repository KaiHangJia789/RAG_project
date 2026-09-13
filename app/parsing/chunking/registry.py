"""
Chunk 策略注册表
按策略名获取切分器实例。
"""
from app.parsing.chunking.base import ChunkStrategy
from app.parsing.chunking.fixed_size import FixedSizeChunker
from app.parsing.chunking.paragraph import ParagraphChunker
from app.parsing.chunking.sentence import SentenceChunker

# 预置策略实例（默认参数）
_REGISTRY: dict[str, ChunkStrategy] = {
    FixedSizeChunker.name: FixedSizeChunker(),
    ParagraphChunker.name: ParagraphChunker(),
    SentenceChunker.name: SentenceChunker(),
}


def get_chunker(name: str) -> ChunkStrategy:
    """
    按策略名获取切分器实例。

    Raises:
        ValueError: 未知策略名
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"未知 Chunk 策略 '{name}'，可用: {available}")
    return _REGISTRY[name]


def list_strategies() -> list[str]:
    """列出所有已注册的策略名"""
    return sorted(_REGISTRY.keys())
