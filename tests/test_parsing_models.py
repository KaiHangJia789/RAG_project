"""
解析数据模型测试
"""
import pytest
from app.parsing.models import (
    BlockType,
    ContentBlock,
    ParsedDocument,
    TableData,
    PdfPageType,
    ChunkingConfig,
)


class TestContentBlock:
    def test_create_paragraph_block(self):
        block = ContentBlock(
            block_type=BlockType.PARAGRAPH,
            text="测试段落",
            page_number=3,
            position=0,
        )
        assert block.block_type == BlockType.PARAGRAPH
        assert block.text == "测试段落"
        assert block.page_number == 3

    def test_create_heading_block(self):
        block = ContentBlock(
            block_type=BlockType.HEADING,
            text="标题",
            position=1,
            metadata={"level": 2},
        )
        assert block.block_type == BlockType.HEADING
        assert block.metadata["level"] == 2

    def test_create_scanned_block(self):
        block = ContentBlock(
            block_type=BlockType.PARAGRAPH,
            text="",
            page_number=4,
            position=2,
            metadata={"needs_ocr": True, "page_type": "scanned"},
        )
        assert block.text == ""
        assert block.metadata["needs_ocr"] is True

    def test_split_block_metadata(self):
        block = ContentBlock(
            block_type=BlockType.PARAGRAPH,
            text="chunk",
            page_number=5,
            position=3,
            metadata={
                "is_split": True,
                "source_page": 5,
                "split_part": 2,
                "split_total": 3,
                "page_number_certain": True,
            },
        )
        assert block.metadata["is_split"] is True
        assert block.metadata["split_part"] == 2


class TestTableData:
    def test_create_table(self):
        table = TableData(
            headers=["姓名", "年龄"],
            rows=[["张三", "28"], ["李四", "32"]],
            page_number=2,
        )
        assert len(table.headers) == 2
        assert len(table.rows) == 2
        assert table.rows[0][0] == "张三"


class TestParsedDocument:
    def test_create_parsed_document(self):
        blocks = [
            ContentBlock(block_type=BlockType.HEADING, text="标题", position=0),
            ContentBlock(block_type=BlockType.PARAGRAPH, text="正文", position=1),
        ]
        doc = ParsedDocument(
            filename="test.pdf",
            file_type=".pdf",
            total_pages=5,
            total_blocks=2,
            blocks=blocks,
            tables=[],
            page_summary={"text_pages": 4, "scanned_pages": 1},
            warnings=["page_3_scanned: requires OCR"],
        )
        assert doc.filename == "test.pdf"
        assert doc.total_pages == 5
        assert doc.total_blocks == 2
        assert len(doc.warnings) == 1
        assert doc.parsed_at is not None


class TestPdfPageType:
    def test_all_types(self):
        assert PdfPageType.TEXT == "text"
        assert PdfPageType.SCANNED == "scanned"
        assert PdfPageType.HYBRID == "hybrid"
        assert PdfPageType.BLANK == "blank"


class TestChunkingConfig:
    def test_default_config(self):
        config = ChunkingConfig()
        assert config.max_chars == 1000
        assert config.overlap_chars == 200
        assert config.merge_short_threshold == 100
        assert "\n\n" in config.split_on

    def test_custom_config(self):
        config = ChunkingConfig(max_chars=500, overlap_chars=100)
        assert config.max_chars == 500
        assert config.overlap_chars == 100
