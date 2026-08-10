"""
纯文本解析器 — 编码检测 + 乱码警告 + 段落分割

核心能力:
  - chardet 自动检测编码（UTF-8 / GBK / ...）
  - 乱码检测：替换字符 U+FFFD 占比 > 5% → warning
  - 按连续空行分割段落
  - 过滤全是空白/控制字符的伪段落
"""
import logging
import re
from datetime import datetime, UTC

from app.parsing.base import BaseParser
from app.parsing.models import (
    BlockType,
    ContentBlock,
    ParsedDocument,
)

logger = logging.getLogger("rag_api.parsing.txt")


class TxtParser(BaseParser):
    supported_types = (".txt",)

    # 乱码判定阈值
    GARBLED_THRESHOLD = 0.05  # U+FFFD 占比 > 5%

    async def parse(self, filename: str, content: bytes) -> ParsedDocument:
        warnings: list[str] = []

        # 1. 编码检测 + 解码
        text, enc_info = self.detect_and_decode(content)

        # 2. 乱码检测
        replacement_count = text.count("�")
        if replacement_count > 0:
            ratio = replacement_count / max(len(text), 1)
            if ratio > self.GARBLED_THRESHOLD:
                warnings.append(
                    f"encoding_quality_poor: {replacement_count} replacement chars "
                    f"({ratio:.1%}), encoding={enc_info['encoding']} "
                    f"confidence={enc_info['confidence']}"
                )

        # 3. 按连续空行分割段落
        blocks: list[ContentBlock] = []
        raw_paragraphs = re.split(r"\n\s*\n", text)

        for para in raw_paragraphs:
            cleaned = para.strip()

            # 跳过空段落
            if not cleaned:
                continue

            # 跳过全空白/控制字符的伪段落
            if self._is_noise(cleaned):
                continue

            blocks.append(ContentBlock(
                block_type=BlockType.PARAGRAPH,
                text=cleaned,
                position=len(blocks),
            ))

        return ParsedDocument(
            filename=filename,
            file_type=".txt",
            total_pages=1,
            total_blocks=len(blocks),
            blocks=blocks,
            tables=[],
            page_summary={"text_pages": 1},
            warnings=warnings,
            metadata={
                "encoding": enc_info["encoding"],
                "encoding_confidence": enc_info["confidence"],
                "original_byte_size": len(content),
                "decoded_char_count": len(text),
                "replacement_char_count": replacement_count,
            },
            parsed_at=datetime.now(UTC),
        )

    @staticmethod
    def _is_noise(text: str, threshold: float = 0.6) -> bool:
        """
        判定文本是否是噪音（不可打印字符占比过高）。

        例如全是制表符、换页符、控制字符的"伪段落"。
        """
        if not text:
            return True
        # 统计可打印字符（包括中文、英文、数字、标点）
        printable = sum(1 for c in text if c.isprintable() or c in "\n\r\t")
        return printable / len(text) < threshold
