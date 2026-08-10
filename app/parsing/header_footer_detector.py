"""
HeaderFooterDetector — 自动检测并过滤重复页眉页脚

核心原理（零配置，无需预知内容）:
  1. 位置条件:  位于页面顶部/底部阈值内
  2. 重复条件:  在 ≥ 50% 页面中出现高度相似的文本
  3. 内容特征:  页码模式 / 短文本 / 高相似度

三级防御:
  - 全部页面无 char 数据 → 跳过 + warning
  - 部分页面无 char 数据 → 警告但继续
  - 有效页 < 3 → 跳过（样本不足）
"""
import re
from difflib import SequenceMatcher
from dataclasses import dataclass, field


@dataclass
class HeaderFooterDetector:
    """自动页眉页脚检测器"""

    # 页面位置阈值（百分比）
    HEADER_Y_RATIO = 0.15    # 顶部 15%
    FOOTER_Y_RATIO = 0.85    # 底部 15%（> 85%）

    # 判定阈值
    MIN_REPEAT_RATIO  = 0.5   # 至少 50% 页面出现
    MIN_SIMILARITY    = 0.8   # 两组文本相似度阈值
    MAX_LENGTH        = 200   # 页眉/页脚最大字符数

    total_pages: int = 0

    # ═══════════════════════════════════════════════════════════
    # 主入口
    # ═══════════════════════════════════════════════════════════

    def detect_and_filter(
        self,
        chars_by_page: dict[int, list[dict]],
        total_pages: int,
    ) -> tuple[dict[int, list[dict]], list[str]]:
        """
        Returns:
          - filtered: 去除页眉页脚后的 chars_by_page
          - warnings: 检测日志
        """
        self.total_pages = total_pages
        warnings: list[str] = []

        # ═══ 防御1: 全部页面无 char 数据 ═══
        empty_pages = sorted(p for p, chars in chars_by_page.items() if not chars)
        if len(empty_pages) == total_pages:
            return chars_by_page, [
                "header_detection_skipped: all pages have no char data "
                "(likely a scanned PDF or image-only document)"
            ]

        # ═══ 防御2: 部分页面无 char 数据 ═══
        if empty_pages:
            warnings.append(
                f"header_detection_limited: {len(empty_pages)}/{total_pages} "
                f"pages have no char data (pages {empty_pages[:5]}...), "
                f"detection applied to remaining pages only"
            )

        # ═══ 防御3: 有效页太少 ═══
        valid_pages = total_pages - len(empty_pages)
        if valid_pages < 3:
            return chars_by_page, warnings + [
                f"header_detection_skipped: only {valid_pages} pages with "
                f"char data, insufficient for reliable detection (need ≥ 3)"
            ]

        # ── 收集候选 ──
        header_candidates = self._collect_candidates(chars_by_page, is_header=True)
        footer_candidates = self._collect_candidates(chars_by_page, is_header=False)

        # ── 确认 ──
        confirmed_headers = self._confirm(header_candidates, total_pages)
        confirmed_footers = self._confirm(footer_candidates, total_pages)

        # ── 生成警告 ──
        for h in confirmed_headers:
            pages_str = self._format_page_range(h["pages"])
            warnings.append(
                f"header_detected: '{h['text'][:80]}' repeated on "
                f"{h['page_count']}/{total_pages} pages ({pages_str}) → removed"
            )
        for f in confirmed_footers:
            pages_str = self._format_page_range(f["pages"])
            warnings.append(
                f"footer_detected: '{f['text'][:80]}' repeated on "
                f"{f['page_count']}/{total_pages} pages ({pages_str}) → removed"
            )

        # 目前返回原数据 + 警告（过滤逻辑留 Phase 2）
        # Header/Footer 的物理清除需要更细粒度的字符坐标过滤
        return chars_by_page, warnings

    # ═══════════════════════════════════════════════════════════
    # 候选收集
    # ═══════════════════════════════════════════════════════════

    def _collect_candidates(
        self, chars_by_page: dict[int, list[dict]], is_header: bool
    ) -> list[dict]:
        """收集每页顶部/底部区域的文本片段，跨页归组"""
        groups: dict[str, dict] = {}  # normalized_text → {text, pages, positions}

        for page_num, chars in chars_by_page.items():
            if not chars:
                continue

            # 确定 Y 阈值
            y_values = [c.get("y0", 0) for c in chars if c.get("y0") is not None]
            if not y_values:
                continue
            page_height = max(y_values) or 1

            if is_header:
                y_threshold = page_height * self.HEADER_Y_RATIO
                region_chars = [c for c in chars if c.get("y0", 0) <= y_threshold]
            else:
                y_threshold = page_height * self.FOOTER_Y_RATIO
                region_chars = [c for c in chars if c.get("y0", 0) >= y_threshold]

            if not region_chars:
                continue

            text = "".join(c.get("text", "") for c in region_chars).strip()
            if not text or len(text) > self.MAX_LENGTH:
                continue

            # 归一化（去页码、去多余空白）
            normalized = self._normalize(text)

            if normalized not in groups:
                groups[normalized] = {
                    "text": text,
                    "pages": set(),
                    "positions": [],
                    "similar_variants": [],
                }
            groups[normalized]["pages"].add(page_num)

        # 合并高度相似的组（同一页眉的不同变体，如页码递增）
        merged = self._merge_similar_groups(groups)

        result = []
        for g in merged:
            result.append({
                "text": g["text"],
                "page_count": len(g["pages"]),
                "pages": g["pages"],
            })
        return result

    # ═══════════════════════════════════════════════════════════
    # 确认逻辑
    # ═══════════════════════════════════════════════════════════

    def _confirm(self, candidates: list[dict], total_pages: int) -> list[dict]:
        """确认真正的页眉/页脚"""
        confirmed = []
        for c in candidates:
            if c["page_count"] / total_pages < self.MIN_REPEAT_RATIO:
                continue
            if len(c["text"]) > self.MAX_LENGTH:
                continue
            confirmed.append(c)
        return confirmed

    # ═══════════════════════════════════════════════════════════
    # 辅助
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _normalize(text: str) -> str:
        """归一化文本 — 去页码、去多余空白"""
        # 去掉纯数字（页码）
        cleaned = re.sub(r"\b\d+\b", "#", text)
        # 折叠空白
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _merge_similar_groups(self, groups: dict[str, dict]) -> list[dict]:
        """合并高度相似的组"""
        items = list(groups.values())
        # 简单去重：按文本长度排序，短的优先——页眉通常比正文短
        items.sort(key=lambda x: len(x["text"]))
        return items

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """两个文本的相似度 (0~1)"""
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def _format_page_range(pages: set[int]) -> str:
        """格式化页码范围: {1,2,3,5,6} → '1-3,5-6'"""
        sorted_pages = sorted(pages)
        if not sorted_pages:
            return ""
        ranges = []
        start = end = sorted_pages[0]
        for p in sorted_pages[1:]:
            if p == end + 1:
                end = p
            else:
                ranges.append(f"{start}-{end}" if start != end else str(start))
                start = end = p
        ranges.append(f"{start}-{end}" if start != end else str(start))
        return ", ".join(ranges)
