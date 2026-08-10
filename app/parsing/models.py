"""
文档解析数据模型 — ParsedDocument / ContentBlock / TableData

这些模型是解析器的标准化输出格式，下游（ChunkSplitter / Chunking / 检索）
统一消费此结构。
"""
from datetime import datetime, UTC
from enum import Enum

from pydantic import BaseModel, Field


# ── 内容块类型 ──

class BlockType(str, Enum):
    PARAGRAPH = "paragraph"
    HEADING   = "heading"
    TABLE     = "table"
    CODE      = "code"
    LIST_ITEM = "list_item"


# ── PDF 页面分类 ──

class PdfPageType(str, Enum):
    TEXT    = "text"       # 正常文本页
    SCANNED = "scanned"    # 扫描件/图片页（需 OCR）
    HYBRID  = "hybrid"     # 文本 + 图片混合
    BLANK   = "blank"      # 真正的空白页


# ── 单个内容块 ──

class ContentBlock(BaseModel):
    """解析出的一个最小内容单元（段落 / 标题 / 表格 / 代码 / 列表项）"""
    block_type: BlockType
    text: str
    page_number: int | None = None
    position: int = 0
    metadata: dict = Field(default_factory=dict)
    # metadata 示例:
    #   段落:   {"font_size": 12, "is_bold": False}
    #   标题:   {"level": 2}
    #   表格:   {"rows": 5, "cols": 3}
    #   代码:   {"language": "python"}
    #   扫描件: {"needs_ocr": True, "image_count": 1}
    #   切分后: {"is_split": True, "source_page": 3, "split_part": 1, "split_total": 3}


# ── 表格数据 ──

class TableData(BaseModel):
    """解析出的表格"""
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    page_number: int | None = None
    caption: str | None = None


# ── 切分配置 ──

class ChunkingConfig(BaseModel):
    """长块切分策略配置"""
    max_chars: int = 1000
    overlap_chars: int = 200
    merge_short_threshold: int = 100
    split_on: tuple[str, ...] = (
        "\n\n",      # 段落边界（最高优先级）
        "\n",        # 行边界
        "。",        # 句号
        "；",        # 分号
        "，",        # 逗号（最低优先级）
    )


# ── 解析器最终输出 ──

class ParsedDocument(BaseModel):
    """解析器完整输出，下游统一消费"""
    filename: str
    file_type: str                          # ".pdf" / ".md" / ".txt"
    total_pages: int = 1
    total_blocks: int = 0
    blocks: list[ContentBlock] = Field(default_factory=list)
    tables: list[TableData] = Field(default_factory=list)
    page_summary: dict[str, int] = Field(default_factory=dict)
    # 例: {"text_pages": 12, "scanned_pages": 3, "hybrid_pages": 1, "blank_pages": 2}
    warnings: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    parsed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
