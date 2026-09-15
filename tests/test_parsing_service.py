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
    async def test_parse_and_persist(self, service, fake_db):
        md = b"# Doc\n\n## Section\n\nContent here.\n\nMore content."
        parsed, chunk_ids = await service.parse_and_persist(
            "test.md", md, "doc-test-001", chunk_strategy="sentence"
        )
        assert parsed.total_blocks >= 3
        # 短文档被 ChunkingService 合并成 1 块 —— 这是「小文档就是一个 chunk」的
        # 合理行为（Week10 统一切分路径后，不再像旧 ChunkSplitter 那样逐段输出）
        assert len(chunk_ids) >= 1
        for cid in chunk_ids:
            assert len(cid) > 0
        # 关键回归：chunk_strategy 标签必须与切分策略一致，
        # 否则检索时会在错误的策略索引里找不到该文档
        chunks = fake_db.table("chunks")
        for cid in chunk_ids:
            assert chunks[cid]["chunk_strategy"] == "sentence"

    @pytest.mark.asyncio
    async def test_parse_txt_and_persist(self, service):
        txt = "段落一\n\n段落二\n\n段落三".encode("utf-8")
        parsed, chunk_ids = await service.parse_and_persist(
            "doc.txt", txt, "doc-test-002"
        )
        assert parsed.total_blocks == 3
        assert len(chunk_ids) >= 1
        for cid in chunk_ids:
            assert len(cid) > 0
