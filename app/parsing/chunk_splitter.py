"""
ChunkSplitter — 将过长的 ContentBlock 按语义边界切分为可控大小的子块

核心设计:
  - 优先在 \n\n（段落边界）切分
  - 找不到边界时依次降级: \n → 。→ ； → ，
  - 相邻块保留 overlap_chars 重叠保持语义连续
  - 不拆分 HEADING 和 CODE 块
  - 跨页段落诚实标记: is_split + source_page + page_number_certain=False
"""
from app.parsing.models import BlockType, ChunkingConfig, ContentBlock


class ChunkSplitter:
    """长块切分器"""

    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or ChunkingConfig()

    # ═══════════════════════════════════════════════════════════
    # 主入口
    # ═══════════════════════════════════════════════════════════

    def split(self, blocks: list[ContentBlock]) -> list[ContentBlock]:
        """
        将超长 ContentBlock 切分为多个子块。

        规则:
          - HEADING / CODE 不拆分
          - 长度 ≤ max_chars 不拆分
          - 超过 max_chars → 按语义边界切分
          - 短于 merge_short_threshold 的子块与上一块合并
        """
        result: list[ContentBlock] = []

        for block in blocks:
            if block.block_type in (BlockType.HEADING, BlockType.CODE):
                result.append(block)
                continue

            if len(block.text) <= self.config.max_chars:
                result.append(block)
                continue

            sub_blocks = self._split_long_block(block)
            result.extend(sub_blocks)

        # 合并过短的相邻块
        result = self._merge_short_blocks(result)

        # 重新编号 position
        for i, b in enumerate(result):
            b.position = i

        return result

    # ═══════════════════════════════════════════════════════════
    # 长块切分
    # ═══════════════════════════════════════════════════════════

    def _split_long_block(self, block: ContentBlock) -> list[ContentBlock]:
        """将一个长块切分为多个子块，诚实标记 page_number 归属"""
        text = block.text
        n = len(text)
        chunk_size = self.config.max_chars
        overlap = self.config.overlap_chars

        sub_blocks: list[ContentBlock] = []
        start = 0
        part_index = 0

        while start < n:
            end = min(start + chunk_size, n)

            # 不是最后一块时 → 找最佳切分点
            if end < n:
                split_point = self._find_best_split_point(text, end)
                if split_point > start:
                    end = split_point

            chunk_text = text[start:end]

            # 诚实标记页码归属
            page_meta = {
                "source_page": block.page_number,
                "is_split": True,
                "split_part": part_index + 1,
                "split_total": 0,  # 回填
                "page_number_certain": True,
                "char_offset_start": start,
                "char_offset_end": end,
            }

            if block.metadata.get("cross_page_boundary"):
                page_meta["cross_page_boundary"] = True
                page_meta["page_number_certain"] = False

            sub_blocks.append(ContentBlock(
                block_type=block.block_type,
                text=chunk_text,
                page_number=block.page_number,
                position=0,
                metadata=page_meta,
            ))

            part_index += 1

            if end >= n:
                break
            start = end - overlap

        # 回填 split_total
        for sb in sub_blocks:
            sb.metadata["split_total"] = part_index

        return sub_blocks

    # ═══════════════════════════════════════════════════════════
    # 切分点查找
    # ═══════════════════════════════════════════════════════════

    def _find_best_split_point(self, text: str, target: int) -> int:
        """
        在 target 位置附近找最佳切分点。

        以 target 为中心，向后搜索 min(200, len(text)-target) 字符，
        按优先级匹配: \n\n > \n > 。 > ； > ，
        """
        search_range = min(200, len(text) - target - 1)
        if search_range <= 0:
            return target

        search_window = text[target:target + search_range]

        for delimiter in self.config.split_on:
            pos = search_window.find(delimiter)
            if pos != -1:
                return target + pos + len(delimiter)

        return target

    # ═══════════════════════════════════════════════════════════
    # 短块合并
    # ═══════════════════════════════════════════════════════════

    def _merge_short_blocks(self, blocks: list[ContentBlock]) -> list[ContentBlock]:
        """将过短的块合并到前一块（不可拆分类型除外）"""
        if not blocks:
            return blocks

        threshold = self.config.merge_short_threshold
        result: list[ContentBlock] = []

        for block in blocks:
            if block.block_type in (BlockType.HEADING, BlockType.CODE):
                result.append(block)
                continue

            if result and len(block.text) < threshold:
                prev = result[-1]
                if prev.block_type not in (BlockType.HEADING, BlockType.CODE):
                    # 合并到前一块
                    result[-1] = ContentBlock(
                        block_type=prev.block_type,
                        text=prev.text + "\n" + block.text,
                        page_number=prev.page_number,
                        position=prev.position,
                        metadata={
                            **prev.metadata,
                            "merged_from": prev.metadata.get("split_part", 0),
                            "merged_to": block.metadata.get("split_part", 0),
                        }
                    )
                    continue

            result.append(block)

        return result
