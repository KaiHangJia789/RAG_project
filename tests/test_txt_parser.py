"""
纯文本解析器测试
"""
import pytest
from app.parsing.txt_parser import TxtParser
from app.parsing.models import BlockType


@pytest.fixture
def parser():
    return TxtParser()


class TestTxtParser:
    @pytest.mark.asyncio
    async def test_parse_utf8(self, parser):
        text = "第一段内容测试。\n\n第二段内容。\n\n第三段也在这里。"
        result = await parser.parse("test.txt", text.encode("utf-8"))
        assert result.file_type == ".txt"
        assert result.total_pages == 1
        assert len(result.blocks) == 3

    @pytest.mark.asyncio
    async def test_all_blocks_are_paragraphs(self, parser):
        text = "段落一\n\n段落二"
        result = await parser.parse("test.txt", text.encode())
        for b in result.blocks:
            assert b.block_type == BlockType.PARAGRAPH

    @pytest.mark.asyncio
    async def test_detect_encoding(self, parser):
        text = "中文测试内容"
        result = await parser.parse("test.txt", text.encode("utf-8"))
        assert result.metadata["encoding"].lower() in ("utf-8", "ascii")

    @pytest.mark.asyncio
    async def test_parse_empty_file(self, parser):
        result = await parser.parse("empty.txt", b"")
        assert result.total_blocks == 0
        assert len(result.warnings) == 0

    @pytest.mark.asyncio
    async def test_parse_whitespace_only(self, parser):
        result = await parser.parse("blank.txt", b"\n\n   \n\n\t\n")
        assert result.total_blocks == 0

    @pytest.mark.asyncio
    async def test_parse_single_paragraph(self, parser):
        text = "只有一段"
        result = await parser.parse("single.txt", text.encode())
        assert len(result.blocks) == 1
        assert result.blocks[0].text == "只有一段"

    @pytest.mark.asyncio
    async def test_metadata_fields(self, parser):
        text = "test"
        content = text.encode("utf-8")
        result = await parser.parse("test.txt", content)
        assert result.metadata["original_byte_size"] == len(content)
        assert result.metadata["decoded_char_count"] == len(text)
        assert "encoding" in result.metadata
