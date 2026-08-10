"""
ParsingService — 解析编排层

职责:
  1. 文件 → 类型检测 → 选解析器 → 解析 → ParsedDocument
  2. 可选：切分长块 → 写入 chunks 表（parse_and_persist）
"""
import logging
import time
from pathlib import Path

import asyncpg

from app.db.connection import Database
from app.parsing.chunk_splitter import ChunkSplitter
from app.parsing.exceptions import ParseFailureError, UnsupportedFileTypeError
from app.parsing.models import ParsedDocument
from app.parsing.parser_registry import ParserRegistry, get_default_registry

logger = logging.getLogger("rag_api.parsing.service")


class ParsingService:
    """
    解析编排服务。

    Usage:
        service = ParsingService(registry, db)
        parsed = await service.parse("doc.pdf", pdf_bytes)
        # 或
        parsed, chunk_ids = await service.parse_and_persist(filename, content, doc_id)
    """

    def __init__(
        self,
        db: Database,
        registry: ParserRegistry | None = None,
        splitter: ChunkSplitter | None = None,
    ):
        self.registry = registry or get_default_registry()
        self.db = db
        self.splitter = splitter or ChunkSplitter()

    # ═══════════════════════════════════════════════════════════
    # parse — 纯解析，不写库
    # ═══════════════════════════════════════════════════════════

    async def parse(self, filename: str, content: bytes) -> ParsedDocument:
        """
        解析文件为 ParsedDocument。

        Raises:
            UnsupportedFileTypeError: 文件类型不支持
            ParseFailureError: 解析过程失败
        """
        ext = Path(filename).suffix.lower()
        parser = self.registry.get(ext)
        if parser is None:
            supported = ", ".join(self.registry.supported_types)
            raise UnsupportedFileTypeError(ext)

        start = time.monotonic()
        try:
            result = await parser.parse(filename, content)
        except Exception as exc:
            raise ParseFailureError(filename, str(exc)) from exc

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "parsed '%s' (%s): %d blocks, %d tables, %d pages, %.0fms",
            filename, ext, result.total_blocks,
            len(result.tables), result.total_pages, elapsed_ms,
        )

        if result.warnings:
            for w in result.warnings:
                logger.warning("  [%s] %s", filename, w)

        return result

    # ═══════════════════════════════════════════════════════════
    # parse_and_persist — 解析 + 切分 + 写入 chunks 表
    # ═══════════════════════════════════════════════════════════

    async def parse_and_persist(
        self, filename: str, content: bytes, document_id: str
    ) -> tuple[ParsedDocument, list[str]]:
        """
        解析文件 → 切分长块 → 写入 chunks 表。

        Returns:
            (parsed_document, chunk_ids)
        """
        # 1. 解析
        parsed = await self.parse(filename, content)

        # 2. 切分长块
        blocks = self.splitter.split(parsed.blocks)

        # 3. 写入 chunks 表（事务）
        chunk_ids: list[str] = []
        async with self.db.transaction() as conn:
            for block in blocks:
                tbl = '"chunks"'
                clean_text = self._sanitize_text(block.text)
                chunk = await conn.fetchrow(
                    f"INSERT INTO {tbl} "
                    f"(document_id, chunk_index, chunk_text, chunk_hash, "
                    f"token_count, page_number) "
                    f"VALUES ($1,$2,$3,$4,$5,$6) RETURNING id",
                    document_id,
                    block.position,
                    clean_text,
                    self._hash_text(clean_text),
                    self._estimate_tokens(clean_text),
                    block.page_number,
                )
                chunk_ids.append(str(chunk["id"]))

        logger.info(
            "persisted '%s': %d blocks → %d chunks",
            filename, parsed.total_blocks, len(chunk_ids),
        )
        return parsed, chunk_ids

    # ═══════════════════════════════════════════════════════════
    # 辅助
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """清理文本中的非法字符（\x00 空字节等），防止 PostgreSQL UTF-8 错误"""
        # 移除空字节
        text = text.replace("\x00", "")
        # 移除其他不可打印的控制字符（保留换行、制表符）
        import re
        text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        return text

    @staticmethod
    def _hash_text(text: str) -> str:
        import hashlib
        return hashlib.sha256(text.encode()).hexdigest()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        粗略估算 token 数量。

        中英文混排估算:
          - 中文字符 ~1.5 token/char
          - 英文单词 ~1.3 token/word
          - 折中：字符数 * 0.5
        """
        return max(1, int(len(text) * 0.5))
