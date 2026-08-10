"""
PDF 解析器 — 基于 pdfplumber

核心能力:
  - 文本提取：逐页提取段落，保持页码
  - 表格提取：extract_tables() → TableData
  - 页面分类：_classify_page() 区分 TEXT/SCANNED/HYBRID/BLANK
  - 扫描件检测：非文本页标记 needs_ocr，不静默丢弃
  - 空白页过滤：< BLANK_THRESHOLD 字符的页跳过
"""
import logging
from datetime import datetime, UTC
from io import BytesIO

import pdfplumber

from app.parsing.base import BaseParser
from app.parsing.header_footer_detector import HeaderFooterDetector
from app.parsing.models import (
    BlockType,
    ContentBlock,
    ParsedDocument,
    PdfPageType,
    TableData,
)

logger = logging.getLogger("rag_api.parsing.pdf")


class PdfParser(BaseParser):
    supported_types = (".pdf",)

    # ── 阈值配置 ──
    MIN_TEXT_CHARS  = 50     # 文本字符数少于此 → 可能为扫描件
    MIN_IMAGE_AREA  = 0.3    # 图片面积占比超过此 → 可能为扫描件
    BLANK_THRESHOLD = 10     # 真正空白页阈值

    def __init__(self, hf_detector: HeaderFooterDetector | None = None):
        self.hf_detector = hf_detector or HeaderFooterDetector()

    # ═══════════════════════════════════════════════════════════
    # 主入口
    # ═══════════════════════════════════════════════════════════

    async def parse(self, filename: str, content: bytes) -> ParsedDocument:
        """解析 PDF 文件为 ParsedDocument"""
        warnings: list[str] = []
        blocks: list[ContentBlock] = []
        tables: list[TableData] = []
        page_summary: dict[str, int] = {
            "text_pages": 0, "scanned_pages": 0, "hybrid_pages": 0, "blank_pages": 0
        }

        with pdfplumber.open(BytesIO(content)) as pdf:
            total_pages = len(pdf.pages)

            for page_num, page in enumerate(pdf.pages, start=1):
                # 1. 页面分类
                page_type = self._classify_page(page)

                if page_type == PdfPageType.BLANK:
                    page_summary["blank_pages"] += 1
                    continue

                if page_type == PdfPageType.SCANNED:
                    page_summary["scanned_pages"] += 1
                    image_count = len(getattr(page, "images", []))
                    blocks.append(ContentBlock(
                        block_type=BlockType.PARAGRAPH,
                        text="",
                        page_number=page_num,
                        position=len(blocks),
                        metadata={
                            "page_type": "scanned",
                            "needs_ocr": True,
                            "image_count": image_count,
                        }
                    ))
                    warnings.append(
                        f"page_{page_num}_scanned: no extractable text, "
                        f"requires OCR ({image_count} images detected)"
                    )
                    continue

                if page_type == PdfPageType.HYBRID:
                    page_summary["hybrid_pages"] += 1
                    warnings.append(
                        f"page_{page_num}_hybrid: text+image page, images skipped"
                    )
                else:
                    page_summary["text_pages"] += 1

                # 2. 提取文本
                text = page.extract_text()
                if text:
                    for para in self._split_paragraphs(text):
                        if not self.is_blank_text(para):
                            blocks.append(ContentBlock(
                                block_type=BlockType.PARAGRAPH,
                                text=para.strip(),
                                page_number=page_num,
                                position=len(blocks),
                            ))

                # 3. 提取表格
                page_tables = page.extract_tables()
                if page_tables:
                    for t_data in page_tables:
                        if not t_data:
                            continue
                        headers = [str(h or "") for h in t_data[0]]
                        rows = [[str(c or "") for c in row] for row in t_data[1:]]
                        if not rows:
                            warnings.append(
                                f"page_{page_num}_table_empty: table with headers "
                                f"but no data rows skipped"
                            )
                            continue
                        table = TableData(
                            headers=headers,
                            rows=rows,
                            page_number=page_num,
                        )
                        tables.append(table)
                        blocks.append(ContentBlock(
                            block_type=BlockType.TABLE,
                            text=self._table_to_text(table),
                            page_number=page_num,
                            position=len(blocks),
                            metadata={"rows": len(rows), "cols": len(headers)}
                        ))

        # 4. 页眉页脚检测（位置 + 重复次数 + 相似度）
        # 这里获取 chars_by_page 对整个文档做一次检测
        chars_by_page = self._extract_chars_by_page(content)
        _filtered, hf_warnings = self.hf_detector.detect_and_filter(
            chars_by_page, total_pages
        )
        warnings.extend(hf_warnings)

        # 5. 组装结果
        return ParsedDocument(
            filename=filename,
            file_type=".pdf",
            total_pages=total_pages,
            total_blocks=len(blocks),
            blocks=blocks,
            tables=tables,
            page_summary=page_summary,
            warnings=warnings,
            metadata={},
            parsed_at=datetime.now(UTC),
        )

    # ═══════════════════════════════════════════════════════════
    # 页面分类
    # ═══════════════════════════════════════════════════════════

    def _classify_page(self, page) -> PdfPageType:
        """
        三级分类逻辑:
          1. 提取文本长度
          2. 检测图片面积占比
          3. 综合判定 → TEXT / SCANNED / HYBRID / BLANK
        """
        text = page.extract_text() or ""
        text_len = len(text.strip())

        images = getattr(page, "images", [])
        image_count = len(images)

        image_ratio = 0.0
        if image_count > 0 and page.width and page.height:
            total_area = page.width * page.height
            image_areas = []
            for img in images:
                w = img.get("x1", 0) - img.get("x0", 0)
                h = img.get("y1", 0) - img.get("y0", 0)
                image_areas.append(abs(w * h))
            total_image_area = sum(image_areas)
            image_ratio = total_image_area / total_area

        # 判定矩阵
        if text_len <= self.BLANK_THRESHOLD and image_count == 0:
            return PdfPageType.BLANK

        if text_len <= self.MIN_TEXT_CHARS and image_count > 0 and image_ratio > self.MIN_IMAGE_AREA:
            return PdfPageType.SCANNED

        if text_len > self.BLANK_THRESHOLD and image_count > 0:
            return PdfPageType.HYBRID

        return PdfPageType.TEXT

    # ═══════════════════════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        """按连续换行符分割段落"""
        import re
        return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    @staticmethod
    def _table_to_text(table: TableData) -> str:
        """表格 → 可检索的文本表示"""
        lines = [" | ".join(table.headers)]
        lines.append(" | ".join("---" for _ in table.headers))
        for row in table.rows:
            lines.append(" | ".join(row))
        return "\n".join(lines)

    @staticmethod
    def _extract_chars_by_page(content: bytes) -> dict[int, list[dict]]:
        """从 PDF bytes 提取每页的字符级数据（供 HeaderFooterDetector 使用）"""
        result: dict[int, list[dict]] = {}
        try:
            with pdfplumber.open(BytesIO(content)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    chars = getattr(page, "chars", None)
                    if chars:
                        result[page_num] = [
                            {"text": c.get("text", ""),
                             "y0": c.get("top", 0),
                             "y1": c.get("bottom", 0)}
                            for c in chars
                        ]
                    else:
                        result[page_num] = []
        except Exception:
            pass
        return result
