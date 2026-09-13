"""
Chunk 切分策略测试
"""
import pytest

from app.parsing.chunking import (
    FixedSizeChunker,
    ParagraphChunker,
    SentenceChunker,
    get_chunker,
    list_strategies,
)
from app.parsing.chunking.comparator import ChunkComparator


class TestFixedSizeChunker:
    def test_basic_split(self):
        chunker = FixedSizeChunker(chunk_size=10, overlap=0)
        chunks = chunker.chunk("0123456789abcdefghij")
        assert len(chunks) == 2
        assert chunks[0].text == "0123456789"
        assert chunks[1].text == "abcdefghij"

    def test_overlap(self):
        chunker = FixedSizeChunker(chunk_size=10, overlap=3)
        chunks = chunker.chunk("0123456789abcdefghij")
        # 第一块 0-9，第二块从 7 开始（overlap 3）
        assert chunks[0].text == "0123456789"
        assert chunks[1].text.startswith("789")

    def test_empty_text(self):
        chunker = FixedSizeChunker()
        assert chunker.chunk("") == []

    def test_text_shorter_than_chunk(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk("short")
        assert len(chunks) == 1

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=0, overlap=0)
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=10, overlap=10)  # overlap >= chunk_size


class TestParagraphChunker:
    def test_basic_split(self):
        chunker = ParagraphChunker()
        chunks = chunker.chunk("第一段\n\n第二段\n\n第三段")
        assert len(chunks) == 3

    def test_empty_text(self):
        assert ParagraphChunker().chunk("") == []

    def test_min_length_filter(self):
        chunker = ParagraphChunker(min_length=3)
        chunks = chunker.chunk("ab\n\n较长的一段文字")
        assert len(chunks) == 1
        assert "较长" in chunks[0].text


class TestSentenceChunker:
    def test_split_by_sentence(self):
        chunker = SentenceChunker(max_chars=500)
        chunks = chunker.chunk("这是第一句。这是第二句！这是第三句？")
        assert len(chunks) == 1  # 都合并到一个块（未超 max_chars）
        assert "第一句" in chunks[0].text

    def test_merge_to_max_chars(self):
        chunker = SentenceChunker(max_chars=20)
        text = "一二三四五六七八九十。一二三四五六七八九十。一二三四五六七八九十。"
        chunks = chunker.chunk(text)
        # 每句 11 字，20 字上限 → 每块约 1-2 句
        assert len(chunks) >= 2

    def test_long_sentence_hard_split(self):
        chunker = SentenceChunker(max_chars=10)
        chunks = chunker.chunk("一二三四五六七八九十一二三四五六七八九十")
        # 无句子分隔符，超长 → 硬切
        assert len(chunks) >= 2

    def test_empty_text(self):
        assert SentenceChunker().chunk("") == []


class TestRegistry:
    def test_list_strategies(self):
        names = list_strategies()
        assert "fixed_size" in names
        assert "paragraph" in names
        assert "sentence" in names

    def test_get_chunker(self):
        assert isinstance(get_chunker("sentence"), SentenceChunker)

    def test_unknown_strategy(self):
        with pytest.raises(ValueError):
            get_chunker("nonexistent")


class TestChunkComparator:
    def test_compare(self):
        comparator = ChunkComparator()
        report = comparator.compare(["这是一段测试文本。用于验证三种切分策略。"])
        assert len(report.results) == 3
        assert report.text_count == 1

    def test_comparison_table_contains_all_strategies(self):
        comparator = ChunkComparator()
        report = comparator.compare(["测试文本。"])
        for name in ["fixed_size", "paragraph", "sentence"]:
            assert name in report.comparison_table
