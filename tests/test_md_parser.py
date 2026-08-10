"""
Markdown 解析器测试
"""
import pytest
from app.parsing.markdown_parser import MarkdownParser
from app.parsing.models import BlockType


@pytest.fixture
def parser():
    return MarkdownParser()


class TestMarkdownParser:
    @pytest.mark.asyncio
    async def test_parse_headings(self, parser):
        md = "# Title\n\n## Section 1\n\nsome text\n\n### Sub\n\nmore text"
        result = await parser.parse("test.md", md.encode())
        headings = [b for b in result.blocks if b.block_type == BlockType.HEADING]
        assert len(headings) == 3
        assert headings[0].metadata["level"] == 1
        assert headings[1].metadata["level"] == 2
        assert headings[2].metadata["level"] == 3

    @pytest.mark.asyncio
    async def test_parse_paragraphs(self, parser):
        md = "# Title\n\n第一段内容。\n\n第二段内容。\n\n第三段。"
        result = await parser.parse("test.md", md.encode())
        paragraphs = [b for b in result.blocks if b.block_type == BlockType.PARAGRAPH]
        assert len(paragraphs) == 3

    @pytest.mark.asyncio
    async def test_parse_code_block(self, parser):
        md = "# Code\n\n```python\nprint('hello')\ndef foo():\n    pass\n```"
        result = await parser.parse("test.md", md.encode())
        code_blocks = [b for b in result.blocks if b.block_type == BlockType.CODE]
        assert len(code_blocks) >= 1
        assert "python" in str(code_blocks[0].metadata.get("language", ""))

    @pytest.mark.asyncio
    async def test_parse_list_items(self, parser):
        md = "# List\n\n- item 1\n- item 2\n  - nested a\n  - nested b\n- item 3"
        result = await parser.parse("test.md", md.encode())
        list_items = [b for b in result.blocks if b.block_type == BlockType.LIST_ITEM]
        assert len(list_items) >= 3

    @pytest.mark.asyncio
    async def test_parse_gfm_table(self, parser):
        md = "| Col A | Col B |\n|-------|-------|\n| a1 | b1 |\n| a2 | b2 |"
        result = await parser.parse("test.md", md.encode())
        tables = [b for b in result.blocks if b.block_type == BlockType.TABLE]
        assert len(tables) >= 1
        assert len(result.tables) >= 1

    @pytest.mark.asyncio
    async def test_parse_metadata(self, parser):
        md = "# T\n\nparagraph text\n\n```py\ncode\n```"
        result = await parser.parse("test.md", md.encode())
        assert result.metadata["heading_count"] == 1
        assert result.metadata["code_block_count"] >= 1

    @pytest.mark.asyncio
    async def test_parse_chinese(self, parser):
        md = "# 中文标题\n\n这是一段中文内容，用于测试中文 Markdown 解析。\n\n## 第二节\n\n更多内容。"
        result = await parser.parse("test.md", md.encode())
        assert result.total_blocks >= 3
        assert len(result.warnings) == 0

    @pytest.mark.asyncio
    async def test_parse_empty(self, parser):
        result = await parser.parse("empty.md", b"")
        assert result.total_blocks == 0
