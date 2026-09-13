"""
固定大小切分策略
按固定字符数硬切，带 overlap（重叠）保持上下文连续性。
"""
from app.parsing.chunking.base import Chunk, ChunkStrategy


class FixedSizeChunker(ChunkStrategy):
    """
    固定大小切分：每 chunk_size 字符切一刀，相邻块重叠 overlap 字符。

    优点：实现简单、可控，chunk 数量可预测。
    缺点：可能在句子/词中间切断，语义不完整。
    """

    name = "fixed_size"

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self._validate_params(chunk_size=chunk_size, overlap=overlap)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[Chunk]:
        if not text:
            return []

        chunks: list[Chunk] = []
        start = 0
        n = len(text)

        while start < n:
            end = min(start + self.chunk_size, n)
            chunks.append(Chunk(text=text[start:end], start=start, end=end))

            if end >= n:
                break
            # 下一块从 end - overlap 开始（重叠区）
            start = end - self.overlap

        return chunks
