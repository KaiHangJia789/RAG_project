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

    def __init__(self, max_chars: int = 500, overlap: int = 0):
        """
        Args:
            max_chars: 目标块最大字符数
            overlap: 相邻块重叠的字符数（0 = 不重叠）
        """
        self._validate_params(chunk_size=max_chars, overlap=overlap)
        self.max_chars = max_chars
        self.overlap = overlap

    def chunk(self, text: str) -> list[Chunk]:
        if not text:
            return []

        sentences = self._split_sentences(text)
        return self._merge_sentences(sentences)

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
