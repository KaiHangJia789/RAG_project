"""
解析编排服务测试
"""
import pytest
from app.parsing.parser_registry import ParserRegistry, get_default_registry
from app.parsing.exceptions import UnsupportedFileTypeError, ParseFailureError
from app.services.parsing_service import ParsingService


@pytest.fixture
def service(fake_db):
    """创建使用内存 DB + 真实解析器的 ParsingService"""
    from app.parsing.chunk_splitter import ChunkSplitter, ChunkingConfig
    registry = get_default_registry()
    # 测试用：merge_short_threshold=0 避免短段落被合并
    splitter = ChunkSplitter(ChunkingConfig(merge_short_threshold=0))
    return ParsingService(db=fake_db, registry=registry, splitter=splitter)


class TestParsingService:
    @pytest.mark.asyncio
    async def test_parse_md(self, service):
        md = b"# Title\n\nparagraph text"
        result = await service.parse("test.md", md)
        assert result.file_type == ".md"
        assert result.total_blocks >= 2

    @pytest.mark.asyncio
    async def test_parse_txt(self, service):
        txt = "段落一\n\n段落二".encode("utf-8")
        result = await service.parse("doc.txt", txt)
        assert result.file_type == ".txt"
        assert result.total_blocks == 2

    @pytest.mark.asyncio
    async def test_unsupported_type(self, service):
        with pytest.raises(UnsupportedFileTypeError):
            await service.parse("image.png", b"fake png")

    @pytest.mark.asyncio
    async def test_unsupported_type_in_message(self, service):
        with pytest.raises(UnsupportedFileTypeError) as exc:
            await service.parse("file.xyz", b"data")
        assert ".xyz" in str(exc.value)

    @pytest.mark.asyncio
    async def test_parse_empty_md(self, service):
        result = await service.parse("empty.md", b"")
        assert result.total_blocks == 0

    @pytest.mark.asyncio
    async def test_parse_and_persist(self, service):
        md = b"# Doc\n\n## Section\n\nContent here.\n\nMore content."
        parsed, chunk_ids = await service.parse_and_persist(
            "test.md", md, "doc-test-001"
        )
        assert parsed.total_blocks >= 3
        assert len(chunk_ids) >= 3
        for cid in chunk_ids:
            assert cid.startswith("doc-") or len(cid) > 0

    @pytest.mark.asyncio
    async def test_parse_txt_and_persist(self, service):
        txt = "段落一\n\n段落二\n\n段落三".encode("utf-8")
        parsed, chunk_ids = await service.parse_and_persist(
            "doc.txt", txt, "doc-test-002"
        )
        assert parsed.total_blocks == 3
        assert len(chunk_ids) == 3
        for cid in chunk_ids:
            assert len(cid) > 0
