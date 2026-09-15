"""
ChunkingService — 面向文档结构的切分编排

为什么需要这一层:
  Week9 的三种策略（fixed_size/paragraph/sentence）只能吃**纯文本**，
  而解析器输出的是**带类型的内容块列表**（标题/段落/代码/表格）。
  直接对整篇文本套策略会丢掉页码、也会把标题和代码块切碎。

  本层负责:
    1. 按块类型分发 —— 标题和代码块不切分，直接透传（保住结构信号）
    2. 对可切的块调用具体策略
    3. **把父块的 page_number 继承给所有子块** —— 引用可定位的前提

参数预设集中在 PARAM_PRESETS，保证"对比不同切分参数"时可复现、可追溯。
"""
import logging

from app.parsing.chunking import build_chunker
from app.parsing.chunking.base import Chunk
from app.parsing.chunk_splitter import ChunkSplitter
from app.parsing.models import BlockType, ChunkingConfig, ContentBlock

logger = logging.getLogger("rag_api.chunking")

# 不可切分的块类型：切了会破坏语义（标题是结构信号，代码切了无法理解）
ATOMIC_BLOCK_TYPES = frozenset({BlockType.HEADING, BlockType.CODE})


# ═══════════════════════════════════════════════════════════════
# 策略参数预设
# ═══════════════════════════════════════════════════════════════
#
# 为了公平对比，三种策略的目标块大小都对齐到 splitter 的 max_chars=1000。
# 差异只在"边界怎么找"，这才是我们要测量的变量。
#
# "splitter" 是线上策略，走原有的 ChunkSplitter（保 overlap=200），
# 其余是 Week10 接入的三种通用策略。

PARAM_PRESETS: dict[str, dict] = {
    "splitter":     {"max_chars": 1000, "overlap": 200},
    "sentence":     {"max_chars": 1000, "overlap": 0},
    "fixed_size":   {"chunk_size": 1000, "overlap": 200},
    # paragraph 的上限刻意比其他策略低：它的设计意图是"尊重段落边界"，
    # 若上限也设 1000，短段落会被合并到与 sentence 完全相同的粒度，失去对比意义。
    # 600 字上限让它真正产出段落级的块（实测 6 块 vs sentence 的 3 块）。
    "paragraph":    {"min_chars": 200, "max_chars": 600},
}

DEFAULT_STRATEGY = "splitter"


def list_presets() -> list[str]:
    """列出所有可用的策略预设名"""
    return sorted(PARAM_PRESETS)


class ChunkingService:
    """把解析出的内容块列表切成带页码的文本块"""

    def __init__(self, strategy: str = DEFAULT_STRATEGY) -> None:
        if strategy not in PARAM_PRESETS:
            raise ValueError(
                f"未知切分策略 '{strategy}'，可用: {', '.join(list_presets())}"
            )
        self.strategy = strategy
        self.params = dict(PARAM_PRESETS[strategy])
        self._chunker = self._build(strategy, self.params)

    @staticmethod
    def _build(strategy: str, params: dict):
        """构造具体切分器"""
        if strategy == "splitter":
            # 线上策略：复用既有的 ChunkSplitter（带 overlap 与语义边界降级）
            return ChunkSplitter(ChunkingConfig(
                max_chars=params["max_chars"],
                overlap_chars=params["overlap"],
            ))
        return build_chunker(strategy, **params)

    # ═══════════════════════════════════════════════════════════
    # 主入口
    # ═══════════════════════════════════════════════════════════

    def chunk_blocks(self, blocks: list[ContentBlock]) -> list[Chunk]:
        """
        把内容块列表切成文本块（保留页码）。

        流程是**两级**的：先按页把相邻块合并成一段文本，再对该文本套用切分策略。

        为什么要先合并：解析器输出的块本身就是段落级的（Markdown 一个自然段
        一个块、平均几十字），逐块套策略的话每个块都远小于目标长度，
        策略根本没机会发挥作用 —— 实测四种策略会产出完全相同的 40 块、
        均长 62 字、碎片率 45%。合并到页码边界后策略才真正在"找边界"。

        为什么以页为界：跨页的块无法归属到确定页码，引用就定位不了。
        页是硬边界，页内才由策略决定怎么切。

        返回的 Chunk.start/end 是**块内偏移**而非全文偏移 —— 跨块拼接后
        没有全局偏移的概念。定位靠 page_number + chunk_index，不靠字符偏移。
        """
        if not blocks:
            return []

        chunks: list[Chunk] = []
        for text, page in self._merge_by_page(blocks):
            chunks.extend(self._chunk_segment(text, page))

        logger.debug(
            "切分完成: 策略=%s, %d 个内容块 → %d 个 chunk",
            self.strategy, len(blocks), len(chunks),
        )
        return chunks

    def chunk_text(self, text: str, page_number: int | None = None) -> list[Chunk]:
        """切分纯文本（无块结构时用，例如脚本直接读 .md 文件）"""
        if not text:
            return []
        return self._chunk_segment(text, page_number)

    # ═══════════════════════════════════════════════════════════
    # 第一级：按页合并
    # ═══════════════════════════════════════════════════════════

    def _merge_by_page(self, blocks: list[ContentBlock]) -> list[tuple[str, int | None]]:
        """
        把相邻块合并成 (文本, 页码) 段落。

        规则：
          - 页码变化 → 断开（保住页码归属）
          - CODE 块独占一段（切了代码就无法理解，也不该和正文混在一起）
          - 其余（标题/段落/列表/表格）用空行拼接
        """
        segments: list[tuple[str, int | None]] = []
        buffer: list[str] = []
        cur_page: int | None = None

        def flush() -> None:
            nonlocal buffer, cur_page
            if buffer:
                segments.append(("\n\n".join(buffer), cur_page))
                buffer = []

        for block in blocks:
            text = (block.text or "").strip()
            if not text:
                continue

            # 代码块独占，且不并入相邻正文
            if block.block_type == BlockType.CODE:
                flush()
                segments.append((text, block.page_number))
                continue

            # 页码变了 → 断开，避免跨页块无法归属
            if buffer and block.page_number != cur_page:
                flush()

            if not buffer:
                cur_page = block.page_number
            buffer.append(text)

        flush()
        return segments

    # ═══════════════════════════════════════════════════════════
    # 第二级：套用切分策略
    # ═══════════════════════════════════════════════════════════

    def _chunk_segment(self, text: str, page: int | None) -> list[Chunk]:
        """对合并后的一段文本套用策略，产出带页码的 Chunk"""
        text = (text or "").strip()
        if not text:
            return []

        if self.strategy == "splitter":
            return self._chunk_with_splitter(text, page)

        return [
            Chunk(text=c.text, start=c.start, end=c.end, page_number=page)
            for c in self._chunker.chunk(text)
            if c.text.strip()
        ]

    def _chunk_with_splitter(self, text: str, page: int | None) -> list[Chunk]:
        """用 ChunkSplitter 切分（它吃 ContentBlock，返回 ContentBlock）"""
        block = ContentBlock(
            block_type=BlockType.PARAGRAPH, text=text, page_number=page
        )
        return [
            Chunk(
                text=sb.text.strip(),
                start=0,
                end=len(sb.text),
                page_number=page,
            )
            for sb in self._chunker.split([block])
            if sb.text and sb.text.strip()
        ]

    # ═══════════════════════════════════════════════════════════
    # 统计（切分质量度量，零成本）
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def quality_metrics(chunks: list[Chunk], *, fragment_threshold: int = 50) -> dict:
        """
        切分质量指标（不需要调用任何模型）。

        fragment_ratio 是重点：碎片率高的策略在检索时会有大量
        低信息量候选干扰排名。
        """
        if not chunks:
            return {
                "count": 0, "avg_len": 0.0, "max_len": 0,
                "min_len": 0, "fragment_ratio": 0.0, "total_chars": 0,
            }

        lens = [c.length for c in chunks]
        fragments = sum(1 for n in lens if n < fragment_threshold)
        return {
            "count": len(chunks),
            "avg_len": round(sum(lens) / len(lens), 1),
            "max_len": max(lens),
            "min_len": min(lens),
            "fragment_ratio": round(fragments / len(chunks), 3),
            "total_chars": sum(lens),
        }
