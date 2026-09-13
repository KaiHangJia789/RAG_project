"""
按段落切分策略
按空行（\\n\\n 或连续换行）边界切分，不切开段落内部。
"""
import re

from app.parsing.chunking.base import Chunk, ChunkStrategy


class ParagraphChunker(ChunkStrategy):
    """
    按段落切分：以连续换行符作为段落边界。

    优点：语义完整，每个段落作为一个整体，不切断句子。
    缺点：段落长度不可控，长段落可能超出 embedding 输入限制。
    """

    name = "paragraph"

    def __init__(self, min_length: int = 1):
        """
        Args:
            min_length: 段落最小长度，短于此的段落被跳过（过滤空段/标题残留）
        """
        self.min_length = max(1, min_length)

    def chunk(self, text: str) -> list[Chunk]:
        if not text:
            return []

        # 按连续换行符切分段落
        paragraphs = re.split(r"\n\s*\n", text)

        chunks: list[Chunk] = []
        pos = 0
        for para in paragraphs:
            # 定位段落起始偏移
            start = text.find(para, pos)
            if start == -1:
                continue
            end = start + len(para)

            cleaned = para.strip()
            if len(cleaned) < self.min_length:
                pos = end
                continue

            chunks.append(Chunk(text=cleaned, start=start, end=end))
            pos = end

        return chunks
