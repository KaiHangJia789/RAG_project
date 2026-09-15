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

    def __init__(
        self,
        min_length: int = 1,
        min_chars: int = 0,
        max_chars: int | None = None,
    ):
        """
        Args:
            min_length: 段落最小长度，短于此的段落被跳过（过滤空段/标题残留）
            min_chars: 合并阈值。累加相邻段落直到长度 >= min_chars 才输出一块。
                默认 0 = 不合并（保持 Week9 行为，既有测试不受影响）。
                Week10 起对比实验用 min_chars=200 —— 否则短段落会切出大量
                平均 31 字符的碎片（Week9 实测 41 块/均长 31），检索必然失败。
            max_chars: 单块长度上限。加上下一段会超限则先输出当前块。
                None = 不设上限。
        """
        if min_chars < 0:
            raise ValueError(f"min_chars 必须 >= 0，实际 {min_chars}")
        if max_chars is not None and max_chars <= 0:
            raise ValueError(f"max_chars 必须 > 0，实际 {max_chars}")
        if max_chars is not None and min_chars > max_chars:
            raise ValueError(
                f"min_chars({min_chars}) 不能大于 max_chars({max_chars})，否则永远凑不满"
            )

        self.min_length = max(1, min_length)
        self.min_chars = min_chars
        self.max_chars = max_chars

    def chunk(self, text: str) -> list[Chunk]:
        if not text:
            return []

        paragraphs = self._split_paragraphs(text)

        # min_chars == 0 且无上限 → 走原路径（逐段输出，行为与 Week9 逐字节一致）
        if self.min_chars == 0 and self.max_chars is None:
            return paragraphs

        return self._merge_paragraphs(paragraphs)

    # ── 段落切分 ──

    def _split_paragraphs(self, text: str) -> list[Chunk]:
        """按连续换行符切分段落"""
        parts = re.split(r"\n\s*\n", text)

        chunks: list[Chunk] = []
        pos = 0
        for para in parts:
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

    # ── 短段落合并 ──

    def _merge_paragraphs(self, paragraphs: list[Chunk]) -> list[Chunk]:
        """
        贪心合并相邻段落。

        规则：
          - 加上下一段会超 max_chars → 先输出当前块，再开新块
          - 当前块长度仍 < min_chars → 继续累加
          - 单个段落本身超 max_chars → 独占一块（不硬切，保住段落语义）
        """
        merged: list[Chunk] = []
        buffer: list[Chunk] = []
        buf_len = 0

        def flush() -> None:
            nonlocal buffer, buf_len
            if buffer:
                merged.append(Chunk(
                    text="\n\n".join(c.text for c in buffer),
                    start=buffer[0].start,
                    end=buffer[-1].end,
                    page_number=buffer[0].page_number,
                ))
                buffer = []
                buf_len = 0

        for para in paragraphs:
            p_len = para.length

            # 单段超上限 → 先把缓冲输出，再让它独占一块
            if self.max_chars is not None and p_len >= self.max_chars:
                flush()
                merged.append(para)
                continue

            # 累加会超上限 → 先输出
            if (
                buffer
                and self.max_chars is not None
                and buf_len + p_len + 2 > self.max_chars   # +2 是拼接用的 "\n\n"
            ):
                flush()

            buffer.append(para)
            buf_len += p_len + 2

        flush()
        return merged
