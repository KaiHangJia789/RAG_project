"""
ChunkSplitter 切分器测试
"""
import pytest
from app.parsing.chunk_splitter import ChunkSplitter, ChunkingConfig
from app.parsing.models import BlockType, ContentBlock


@pytest.fixture
def splitter():
    return ChunkSplitter(ChunkingConfig(
        max_chars=100, overlap_chars=20, merge_short_threshold=30
    ))


@pytest.fixture
def no_merge_splitter():
    return ChunkSplitter(ChunkingConfig(
        max_chars=100, overlap_chars=0, merge_short_threshold=0
    ))


class TestChunkSplitter:
    def test_short_block_not_split(self, splitter):
        block = ContentBlock(block_type=BlockType.PARAGRAPH, text="short", position=0)
        result = splitter.split([block])
        assert len(result) == 1

    def test_long_block_is_split(self, no_merge_splitter):
        text = "这是测试内容。" * 30  # ~210 chars
        block = ContentBlock(block_type=BlockType.PARAGRAPH, text=text, position=0)
        result = no_merge_splitter.split([block])
        assert len(result) >= 2

    def test_heading_not_split(self, splitter):
        text = "# " + "长标题 " * 100
        block = ContentBlock(block_type=BlockType.HEADING, text=text, position=0)
        result = splitter.split([block])
        assert len(result) == 1

    def test_code_not_split(self, splitter):
        text = "code " * 100
        block = ContentBlock(block_type=BlockType.CODE, text=text, position=0)
        result = splitter.split([block])
        assert len(result) == 1

    def test_split_preserves_page_number(self, splitter):
        text = "长文本测试，" * 50
        block = ContentBlock(
            block_type=BlockType.PARAGRAPH, text=text,
            page_number=5, position=0,
        )
        result = splitter.split([block])
        for b in result:
            assert b.page_number == 5
            assert b.metadata["is_split"] is True
            assert b.metadata["source_page"] == 5

    def test_split_metadata_fields(self, no_merge_splitter):
        text = "长文本测试。" * 50
        block = ContentBlock(
            block_type=BlockType.PARAGRAPH, text=text, position=0,
        )
        result = no_merge_splitter.split([block])
        assert len(result) >= 3
        for b in result:
            assert "split_part" in b.metadata
            assert "split_total" in b.metadata

    def test_positions_renumbered(self, splitter):
        text = "长文本。" * 50
        block = ContentBlock(
            block_type=BlockType.PARAGRAPH, text=text, position=5,
        )
        result = splitter.split([block])
        for i, b in enumerate(result):
            assert b.position == i

    def test_short_blocks_merged(self, splitter):
        config = ChunkingConfig(max_chars=1000, merge_short_threshold=50)
        s = ChunkSplitter(config)
        blocks = [
            ContentBlock(block_type=BlockType.PARAGRAPH, text="hi", position=0),
            ContentBlock(block_type=BlockType.PARAGRAPH, text="hello", position=1),
        ]
        result = s.split(blocks)
        assert len(result) == 1
