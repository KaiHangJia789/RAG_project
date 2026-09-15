"""
按句子（语义）切分策略
按句子边界（中英文标点）切分，再把短句合并到接近 max_chars。
这是 Week 7 ChunkSplitter 语义逻辑的重构版。
"""
import re

from app.parsing.chunking.base import Chunk, ChunkStrategy

# 句子结束标点（中文 + 英文）
_SENTENCE_END = "。！？!?"


class SentenceChunker(ChunkStrategy):
    """
    按句子边界切分，再贪心合并到 max_chars。

    优点：语义最完整，块边界落在句子结束处，检索效果好。
    缺点：实现最复杂，块长度不完全可控。
    """

    name = "sentence"

    def __init__(
        self,
        max_chars: int = 500,
        overlap: int = 0,
        split_on: tuple[str, ...] = ("\n\n", "\n"),
    ):
        """
        Args:
            max_chars: 目标块最大字符数
            overlap: 相邻块重叠的字符数（0 = 不重叠）
            split_on: 保底切分边界，按顺序优先匹配。默认先按段落/换行切，
                再在段内按句号切 —— 纯按句号切会把 Markdown 标题、代码块、
                列表项切碎（Week9 实测：同一份文档按句号切成 5 块均长 265，
                但混入了标题残片）。传空元组则回退到纯句号切分。
        """
        self._validate_params(chunk_size=max_chars, overlap=overlap)
        self.max_chars = max_chars
        self.overlap = overlap
        self.split_on = tuple(split_on)

    def chunk(self, text: str) -> list[Chunk]:
        """
        两级切分：先按保底边界（段落/换行）切出结构单元，
        再把结构单元**带分隔符**贪心合并到 max_chars，超长的单元才降级到句级切分。

        为什么不是"把所有句子的句子拉平成一个列表再合并"：那样会丢掉段落
        之间的换行，标题会和正文无缝粘成 "原理与架构什么是 RAG" 这种噪声文本。
        """
        if not text:
            return []

        segments = self._split_by_boundary(text) if self.split_on else [text]

        chunks: list[Chunk] = []
        buffer: list[str] = []
        buf_len = 0

        def flush() -> None:
            nonlocal buffer, buf_len
            if buffer:
                merged = "\n\n".join(buffer)
                chunks.append(Chunk(text=merged, start=0, end=len(merged)))
                buffer = []
                buf_len = 0

        for seg in segments:
            seg_len = len(seg)

            # 单个单元超长 → 先输出缓冲，再按句子切它
            if seg_len > self.max_chars:
                flush()
                chunks.extend(self._merge_sentences(self._split_sentences(seg)))
                continue

            # 累加会超上限 → 先输出
            if buffer and buf_len + seg_len + 2 > self.max_chars:
                flush()

            buffer.append(seg)
            buf_len += seg_len + 2      # +2 是拼接用的 "\n\n"

        flush()
        return chunks

    # ── 保底边界切分 ──

    def _split_by_boundary(self, text: str) -> list[str]:
        """按 split_on 里优先级最高的、实际出现的边界切分（只切一级，不递归）"""
        for delimiter in self.split_on:
            if delimiter in text:
                parts = text.split(delimiter)
                return [p for p in parts if p.strip()]
        return [text]

    # ── 句子切分 ──

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """按句子结束标点切分，保留标点在前一句"""
        # lookbehind 在标点后切分，\s* 吃掉后续空白
        parts = re.split(rf"(?<=[{_SENTENCE_END}])\s*", text)
        return [p.strip() for p in parts if p.strip()]

    # ── 句子合并 ──

    def _merge_sentences(self, sentences: list[str]) -> list[Chunk]:
        chunks: list[Chunk] = []
        current_parts: list[str] = []
        current_len = 0
        offset = 0

        for sentence in sentences:
            s_len = len(sentence)

            # 单个句子超长 → 先 flush 当前累积，再硬切该句子
            if s_len > self.max_chars:
                if current_parts:
                    text = "".join(current_parts)
                    chunks.append(Chunk(text=text, start=offset, end=offset + len(text)))
                    offset += len(text)
                    current_parts = []
                    current_len = 0
                for sub in self._hard_split(sentence):
                    chunks.append(Chunk(text=sub, start=offset, end=offset + len(sub)))
                    offset += len(sub)
                continue

            # 合并后超长 → 先 flush 再开新块
            if current_parts and current_len + s_len > self.max_chars:
                text = "".join(current_parts)
                chunks.append(Chunk(text=text, start=offset, end=offset + len(text)))
                offset += len(text)
                current_parts = []
                current_len = 0

            current_parts.append(sentence)
            current_len += s_len

        # flush 最后剩余
        if current_parts:
            text = "".join(current_parts)
            chunks.append(Chunk(text=text, start=offset, end=offset + len(text)))

        return chunks

    def _hard_split(self, text: str) -> list[str]:
        """硬切超长句子（按 max_chars 固定切）"""
        result = []
        n = len(text)
        start = 0
        while start < n:
            end = min(start + self.max_chars, n)
            result.append(text[start:end])
            start = end
        return result
