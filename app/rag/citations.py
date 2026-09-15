"""
引用处理 —— 上下文装配、引用标记解析、拒答识别

LLM 输出的 [1][2] 编号必须能映射回真实检索结果，否则"引用可点击"就是假的。
本模块负责这条链路的双向转换与校验。

（从 app/llm/experiments.py 的 _has_citation 迁移而来，那边保留别名以兼容既有测试。）
"""
import re

# 引用标记：[1] [12] ［1］（兼容全角括号）
_CITATION_RE = re.compile(r"[\[［](\d{1,3})[\]］]")

# 拒答的常见措辞（LLM 不一定逐字使用 prompt 里指定的句子）
_REFUSAL_MARKERS = (
    "无法回答", "无法确定", "无法从", "没有提供", "未提供", "资料中没有",
    "上下文中没有", "上下文中未", "不足以回答", "不能回答", "无从得知",
    "没有相关信息", "未提及", "无法得知", "不知道",
)

# 引用存在性的宽松标记（_has_citation 用）
_CITATION_HINTS = ("[1]", "[2]", "来源", "reference", "Reference", "引用", "参见")


# ═══════════════════════════════════════════════════════════════
# 上下文装配
# ═══════════════════════════════════════════════════════════════

def build_context(hits, *, max_chars: int = 6000) -> str:
    """
    把检索结果装配成带编号的上下文文本。

    格式（编号是 LLM 引用与 citations 数组对齐的唯一依据）：
        [1] 来源：文件名 · 第 3 页
        正文...
        ---
        [2] 来源：另一文件名
        正文...

    Args:
        hits: IndexHit 列表（须已按相似度降序）
        max_chars: 上下文总长上限，超出则截断尾部（防止撑爆 LLM 输入）
    """
    parts: list[str] = []
    used = 0

    for i, hit in enumerate(hits, start=1):
        rec = hit.record
        source = rec.filename
        if rec.page_number:
            source += f" · 第 {rec.page_number} 页"
        block = f"[{i}] 来源：{source}\n{rec.text}"

        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)

    return "\n---\n".join(parts)


# ═══════════════════════════════════════════════════════════════
# 引用解析
# ═══════════════════════════════════════════════════════════════

def extract_citation_indices(text: str) -> list[int]:
    """抽取答案中出现的引用编号（去重后按首次出现顺序返回）"""
    seen: list[int] = []
    for m in _CITATION_RE.finditer(text or ""):
        idx = int(m.group(1))
        if idx not in seen:
            seen.append(idx)
    return seen


def validate_citations(text: str, hit_count: int) -> tuple[float, list[int]]:
    """
    校验引用编号的有效性。

    Returns:
        (有效率, 越界编号列表)。无引用标记时返回 (1.0, [])，
        因为"没引用"由 citation_coverage 单独衡量，不该在这里算作无效。

    这是零 API 成本、纯正则的幻觉检测 —— 能抓到"编造引用编号"这类
    判官抓不到的幻觉（判官只看语义，不看编号是否真实存在）。
    """
    indices = extract_citation_indices(text)
    if not indices:
        return 1.0, []

    invalid = [i for i in indices if i < 1 or i > hit_count]
    valid_count = len(indices) - len(invalid)
    return valid_count / len(indices), invalid


def strip_invalid_citations(text: str, hit_count: int) -> str:
    """把越界的引用标记从答案里去掉（保留正文，避免前端渲染出死链）"""
    def _sub(m: re.Match) -> str:
        idx = int(m.group(1))
        return m.group(0) if 1 <= idx <= hit_count else ""

    return _CITATION_RE.sub(_sub, text or "")


# ═══════════════════════════════════════════════════════════════
# 拒答识别
# ═══════════════════════════════════════════════════════════════

def is_refusal_lexical(text: str) -> bool:
    """
    词法层面的拒答识别（零成本启发式）。

    只用于快速判断与前端展示；正式评测以判官的 is_refusal 为准
    （判官能识别"换了个说法的拒答"，词法匹配做不到）。
    """
    if not text or not text.strip():
        return True
    body = text.strip()
    # 拒答通常很短；长答案里出现"不知道"多半是在陈述别的内容
    if len(body) > 120:
        return False
    return any(marker in body for marker in _REFUSAL_MARKERS)


def has_citation(text: str) -> bool:
    """答案是否带有引用痕迹（实验指标用，宽松匹配）"""
    return any(hint in (text or "") for hint in _CITATION_HINTS)
