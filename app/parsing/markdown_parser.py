"""
Markdown 解析器 — 基于 mistune v3

核心能力:
  - GFM 表格解析（无需手写正则）
  - 嵌套列表 / 任务列表 / 定义列表
  - 代码块语言识别
  - 标题层级保留
"""
import logging
from datetime import datetime, UTC

import mistune

from app.parsing.base import BaseParser
from app.parsing.models import (
    BlockType,
    ContentBlock,
    ParsedDocument,
    TableData,
)

logger = logging.getLogger("rag_api.parsing.md")


# ═══════════════════════════════════════════════════════════════
# mistune v3 BlockRenderer
# v3 API: token["attrs"] 获取属性, render_children() 渲染子节点
# ═══════════════════════════════════════════════════════════════

class BlockRenderer(mistune.BaseRenderer):
    """将 Markdown AST token 转为 ContentBlock 列表"""

    NAME = "block_collector"

    def __init__(self):
        self.blocks: list[ContentBlock] = []
        self.tables: list[TableData] = []
        self._pos = 0
        super().__init__()

    # ── 辅助：渲染子节点为纯文本 ──

    def _render_children(self, token, state):
        """mistune v3 兼容：递归渲染所有子 token 为字符串"""
        return "".join(
            self.render_token(child, state)
            for child in token.get("children", [])
        )

    # ── 块级元素 ──

    def heading(self, token, state):
        text = self._render_children(token, state)
        level = token["attrs"]["level"]
        self.blocks.append(ContentBlock(
            block_type=BlockType.HEADING,
            text=text,
            position=self._pos,
            metadata={"level": level}
        ))
        self._pos += 1
        return ""

    def paragraph(self, token, state):
        text = self._render_children(token, state)
        if text.strip():
            self.blocks.append(ContentBlock(
                block_type=BlockType.PARAGRAPH,
                text=text,
                position=self._pos,
            ))
            self._pos += 1
        return ""

    def block_code(self, token, state):
        raw = token.get("raw", "")
        if raw:
            code = raw
        else:
            code = self._render_children(token, state)
        language = token.get("attrs", {}).get("info", "unknown")
        self.blocks.append(ContentBlock(
            block_type=BlockType.CODE,
            text=code,
            position=self._pos,
            metadata={"language": language}
        ))
        self._pos += 1
        return ""

    def list(self, token, state):
        """容器 token — 渲染子节点（<ul>/<ol> 包装器）"""
        return self._render_children(token, state)

    def list_item(self, token, state):
        text = self._render_children(token, state)
        attrs = token.get("attrs", {})
        ordered = attrs.get("ordered", False)
        level = attrs.get("level", 1)
        self.blocks.append(ContentBlock(
            block_type=BlockType.LIST_ITEM,
            text=text,
            position=self._pos,
            metadata={"ordered": ordered, "level": level}
        ))
        self._pos += 1
        return ""

    def table(self, token, state):
        header, header_rows = self._extract_table_part(token, "table_head")
        body_text, body_rows = self._extract_table_part(token, "table_body")

        # 如果没有表头，用第一行正文当表头
        if not header and body_rows:
            header = body_rows[0]
            body_rows = body_rows[1:]

        table = TableData(
            headers=header,
            rows=body_rows,
        )
        self.tables.append(table)
        self.blocks.append(ContentBlock(
            block_type=BlockType.TABLE,
            text=Marker._table_to_text(table),
            position=self._pos,
            metadata={"rows": len(table.rows), "cols": len(table.headers)}
        ))
        self._pos += 1
        return ""

    def block_quote(self, token, state):
        text = self._render_children(token, state)
        if text.strip():
            self.blocks.append(ContentBlock(
                block_type=BlockType.PARAGRAPH,
                text=text,
                position=self._pos,
                metadata={"is_quote": True}
            ))
            self._pos += 1
        return ""

    # ── 行内元素（纯文本渲染，不做分块） ──

    def text(self, token, state):
        return token.get("raw", "")

    def codespan(self, token, state):
        return token.get("raw", "")

    def link(self, token, state):
        return self._render_children(token, state)

    def image(self, token, state):
        return token.get("attrs", {}).get("alt", "[image]")

    def strong(self, token, state):
        return self._render_children(token, state)

    def emphasis(self, token, state):
        return self._render_children(token, state)

    def inline_html(self, token, state):
        return ""

    def linebreak(self, token, state):
        return "\n"

    def softbreak(self, token, state):
        return " "

    def blank_line(self, token, state):
        return ""

    def thematic_break(self, token, state):
        return ""

    def block_text(self, token, state):
        return self._render_children(token, state)

    # ── 辅助 ──

    def _extract_table_part(self, token, part_name):
        """提取 table_head 或 table_body 的文本和行数据"""
        for child in token.get("children", []):
            if child.get("type") == part_name:
                rows = []
                for row_token in child.get("children", []):
                    cells = []
                    for cell in row_token.get("children", []):
                        cells.append(self._render_token_only(cell))
                    rows.append(cells)
                if rows:
                    header_text = rows[0]
                    data_rows = rows[1:] if len(rows) > 1 else []
                    return header_text, data_rows
        return [], []

    def _render_token_only(self, token):
        """渲染单个行内 token 为纯文本"""
        if token.get("type") == "text":
            return token.get("raw", "")
        children = token.get("children", [])
        if children:
            return "".join(self._render_token_only(c) for c in children)
        return token.get("raw", "")


# ═══════════════════════════════════════════════════════════════
# MarkdownParser
# ═══════════════════════════════════════════════════════════════

class Marker:
    """静态工具方法"""

    @staticmethod
    def _table_to_text(table: TableData) -> str:
        """表格 → 可检索文本"""
        parts = [" | ".join(table.headers)]
        parts.append(" | ".join("---" for _ in table.headers))
        for row in table.rows:
            parts.append(" | ".join(row))
        return "\n".join(parts)


class MarkdownParser(BaseParser):
    supported_types = (".md",)

    def __init__(self):
        self._markdown = mistune.create_markdown(
            renderer=BlockRenderer(),
            plugins=["table", "task_lists", "def_list"],
        )

    async def parse(self, filename: str, content: bytes) -> ParsedDocument:
        text, enc_info = self.detect_and_decode(content)
        renderer = BlockRenderer()

        md = mistune.create_markdown(
            renderer=renderer,
            plugins=["table", "task_lists", "def_list"],
        )
        md(text)

        blocks = renderer.blocks
        tables = renderer.tables

        heading_count = sum(1 for b in blocks if b.block_type == BlockType.HEADING)
        code_count = sum(1 for b in blocks if b.block_type == BlockType.CODE)

        return ParsedDocument(
            filename=filename,
            file_type=".md",
            total_pages=1,
            total_blocks=len(blocks),
            blocks=blocks,
            tables=tables,
            page_summary={"text_pages": 1},
            warnings=[],
            metadata={
                "encoding": enc_info["encoding"],
                "encoding_confidence": enc_info["confidence"],
                "heading_count": heading_count,
                "code_block_count": code_count,
                "has_gfm_tables": len(tables) > 0,
            },
            parsed_at=datetime.now(UTC),
        )
