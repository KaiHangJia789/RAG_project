"""
Chunk 策略注册表
按策略名获取切分器实例。
"""
from app.parsing.chunking.base import ChunkStrategy
from app.parsing.chunking.fixed_size import FixedSizeChunker
from app.parsing.chunking.paragraph import ParagraphChunker
from app.parsing.chunking.sentence import SentenceChunker

# 策略名 → 类（供 build_chunker 按自定义参数构造）
_CHUNKER_CLASSES: dict[str, type[ChunkStrategy]] = {
    FixedSizeChunker.name: FixedSizeChunker,
    ParagraphChunker.name: ParagraphChunker,
    SentenceChunker.name: SentenceChunker,
}

# 预置策略实例（默认参数）
_REGISTRY: dict[str, ChunkStrategy] = {
    name: cls() for name, cls in _CHUNKER_CLASSES.items()
}


def get_chunker(name: str) -> ChunkStrategy:
    """
    按策略名获取切分器实例（默认参数的**共享单例**）。

    需要自定义 chunk_size/overlap 等参数时用 build_chunker()。

    Raises:
        ValueError: 未知策略名
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"未知 Chunk 策略 '{name}'，可用: {available}")
    return _REGISTRY[name]


def build_chunker(name: str, **params) -> ChunkStrategy:
    """
    按策略名 + 自定义参数构造**新实例**。

    get_chunker 返回的是共享单例，只能拿默认参数；做"不同切分参数"的对比
    实验必须能构造非默认实例（例如让 fixed_size 对齐 splitter 的 1000/200）。

    Raises:
        ValueError: 未知策略名
    """
    if name not in _CHUNKER_CLASSES:
        available = ", ".join(sorted(_CHUNKER_CLASSES.keys()))
        raise ValueError(f"未知 Chunk 策略 '{name}'，可用: {available}")
    return _CHUNKER_CLASSES[name](**params)


def list_strategies() -> list[str]:
    """列出所有已注册的策略名"""
    return sorted(_REGISTRY.keys())
