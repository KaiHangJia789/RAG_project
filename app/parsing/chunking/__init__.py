"""Chunk 切分策略模块 — 3 种策略（固定大小/按段落/按句子）"""
from app.parsing.chunking.base import Chunk, ChunkStrategy
from app.parsing.chunking.fixed_size import FixedSizeChunker
from app.parsing.chunking.paragraph import ParagraphChunker
from app.parsing.chunking.sentence import SentenceChunker
from app.parsing.chunking.registry import build_chunker, get_chunker, list_strategies
from app.parsing.chunking.comparator import ChunkComparator

__all__ = [
    "Chunk",
    "ChunkStrategy",
    "FixedSizeChunker",
    "ParagraphChunker",
    "SentenceChunker",
    "ChunkComparator",
    "build_chunker",
    "get_chunker",
    "list_strategies",
]
