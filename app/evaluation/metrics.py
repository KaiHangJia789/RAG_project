"""
评估指标 — 纯函数，零 IO，零 LLM 调用

所有函数都是确定性的：给定输入必然得到相同输出，可手算验证。
需要调用 LLM 的部分（断言拆解、相关性判定）在 judge.py，本模块只负责
「拿到判定结果之后怎么算分」—— 这样指标口径本身可以被单元测试完整覆盖。

指标口径见 docs/week10/README.md 的「评估指标定义」章节。
"""
from pydantic import BaseModel, Field


class MetricResult(BaseModel):
    """单个指标的取值与解释"""

    name: str
    value: float
    detail: str = ""
    extras: dict = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# Faithfulness（忠实度）— 断言级
# ═══════════════════════════════════════════════════════════════

def faithfulness(
    supported: int,
    total_claims: int,
    *,
    is_refusal: bool = False,
) -> MetricResult:
    """
    忠实度 = 被上下文支持的断言数 / 断言总数

    边界：
      - 拒答 → 1.0（拒答不含事实断言，天然忠实，不该被惩罚）
      - 断言数为 0 且非拒答 → 1.0，但标 degenerate（答案没有事实内容）
    """
    if is_refusal:
        return MetricResult(
            name="faithfulness", value=1.0, detail="拒答，记为完全忠实"
        )
    if total_claims <= 0:
        return MetricResult(
            name="faithfulness", value=1.0,
            detail="无事实断言（退化答案）",
            extras={"degenerate": True},
        )
    if supported < 0 or supported > total_claims:
        raise ValueError(
            f"supported({supported}) 必须在 [0, {total_claims}] 范围内"
        )

    value = supported / total_claims
    return MetricResult(
        name="faithfulness",
        value=round(value, 4),
        detail=f"{supported}/{total_claims} 条断言有上下文支持",
        extras={"supported": supported, "total_claims": total_claims},
    )


# ═══════════════════════════════════════════════════════════════
# Context Precision（上下文精确率）— 排序质量
# ═══════════════════════════════════════════════════════════════

def context_precision_ranked(relevance: list[int | bool]) -> MetricResult:
    """
    Rank-weighted Average Precision（RAGAS v0.1 口径）。

    公式:
        precision@k = (前 k 个中相关数) / k
        CP = Σ(precision@k × relevant_k) / 相关总数

    手算校验（单元测试用的就是这个例子）:
        相关性 [1,0,1,0,0]
        p@1 = 1/1     = 1.0     relevant → 贡献 1.0
        p@2 = 1/2     = 0.5     不相关   → 贡献 0
        p@3 = 2/3     ≈ 0.6667  relevant → 贡献 0.6667
        分子 = 1.6667，分母 = 2 → CP = 0.8333

    分母为 0（检索结果全部不相关）时返回 0.0，不是 NaN。
    """
    rels = [1 if r else 0 for r in relevance]
    if not rels:
        return MetricResult(
            name="context_precision", value=0.0, detail="无检索结果"
        )

    total_relevant = sum(rels)
    if total_relevant == 0:
        return MetricResult(
            name="context_precision", value=0.0,
            detail=f"检索 {len(rels)} 条，全部不相关",
            extras={"retrieved": len(rels), "relevant": 0},
        )

    numerator = 0.0
    running = 0
    for k, r in enumerate(rels, start=1):
        running += r
        if r:
            numerator += running / k

    value = numerator / total_relevant
    return MetricResult(
        name="context_precision",
        value=round(value, 4),
        detail=f"{total_relevant}/{len(rels)} 条相关，rank-weighted AP",
        extras={"retrieved": len(rels), "relevant": total_relevant},
    )


# ═══════════════════════════════════════════════════════════════
# 幻觉率 — 答案级二值
# ═══════════════════════════════════════════════════════════════

def hallucination_rate(answers_with_claims: int, hallucinated: int) -> MetricResult:
    """
    幻觉率 = 含幻觉的答案数 / 有事实断言的答案总数

    定义：答案中存在 ≥1 条无支撑断言即视为「含幻觉」。

    与忠实度的关系是互补而非取反：忠实度是断言级连续量，会被长答案稀释
    （20 条断言里 1 条无支撑 → 忠实度 0.95）；幻觉率按答案计数（同样的答案
    算作 100% 含幻觉）。两个一起看才能同时掌握平均质量与个体风险。

    分母排除拒答与零断言的答案 —— 这两类不可能有幻觉，计入会稀释指标。
    """
    if answers_with_claims <= 0:
        return MetricResult(
            name="hallucination_rate", value=0.0,
            detail="无可评估答案（全部为拒答或零断言）",
        )
    if hallucinated < 0 or hallucinated > answers_with_claims:
        raise ValueError(
            f"hallucinated({hallucinated}) 必须在 [0, {answers_with_claims}] 范围内"
        )

    value = hallucinated / answers_with_claims
    return MetricResult(
        name="hallucination_rate",
        value=round(value, 4),
        detail=f"{hallucinated}/{answers_with_claims} 个答案含无支撑断言",
        extras={"hallucinated": hallucinated, "evaluated": answers_with_claims},
    )


# ═══════════════════════════════════════════════════════════════
# 基于标注来源的确定性检索指标（零 LLM 成本）
# ═══════════════════════════════════════════════════════════════

def hit_rate_at_k(
    retrieved_docs: list[str], reference_docs: set[str]
) -> MetricResult:
    """
    top_k 结果中是否命中任一标注来源文档。

    这是最直观的召回指标，也是自检语料质量的关键工具：
    如果 hit_rate 很低，问题多半在语料或标注，而不在检索代码。
    """
    if not reference_docs:
        return MetricResult(
            name="hit_rate", value=0.0, detail="该题无标注来源（负样本）"
        )
    hit = any(d in reference_docs for d in retrieved_docs)
    return MetricResult(
        name="hit_rate",
        value=1.0 if hit else 0.0,
        detail=f"{'命中' if hit else '未命中'}标注来源（检索 {len(retrieved_docs)} 条）",
    )


def mrr(retrieved_docs: list[str], reference_docs: set[str]) -> MetricResult:
    """
    Mean Reciprocal Rank — 第一个命中标注来源的位置的倒数。

    与 hit_rate 的区别：hit_rate 只看有没有，MRR 还看排在第几位。
    """
    if not reference_docs:
        return MetricResult(name="mrr", value=0.0, detail="该题无标注来源")

    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in reference_docs:
            return MetricResult(
                name="mrr", value=round(1.0 / rank, 4),
                detail=f"首个命中位于第 {rank} 位",
            )
    return MetricResult(name="mrr", value=0.0, detail="未命中任何标注来源")


def source_precision(
    retrieved_docs: list[str], reference_docs: set[str]
) -> MetricResult:
    """
    精确率：top_k 中来自标注来源的比例。

    注意与 context_precision_ranked 的区别：本函数只比较「文档名是否在标注集合里」，
    零 LLM 成本；后者由判官逐条判定相关性，能识别「同一文档但无关的段落」。
    两者结论矛盾时值得深究——要么判官不可靠，要么标注粒度不够。
    """
    if not retrieved_docs:
        return MetricResult(name="source_precision", value=0.0, detail="无检索结果")
    matched = sum(1 for d in retrieved_docs if d in reference_docs)
    return MetricResult(
        name="source_precision",
        value=round(matched / len(retrieved_docs), 4),
        detail=f"{matched}/{len(retrieved_docs)} 条来自标注来源",
    )


# ═══════════════════════════════════════════════════════════════
# 拒答与引用
# ═══════════════════════════════════════════════════════════════

def refusal_accuracy(refused_flags: list[bool], answerable_flags: list[bool]) -> MetricResult:
    """
    拒答正确率 = 正确拒答数 / 应拒答数。

    「应拒答」的定义值得说明：这里统计的是 answerable=False 的题目中被拒答的比例。
    好的 RAG 系统既要能拒答不可回答的问题，也要**不能拒答**可回答的问题，
    后者由「误拒率」单独衡量。
    """
    should_refuse = [i for i, ok in enumerate(answerable_flags) if not ok]
    if not should_refuse:
        return MetricResult(name="refusal_accuracy", value=0.0, detail="无负样本")

    correct = sum(1 for i in should_refuse if refused_flags[i])
    return MetricResult(
        name="refusal_accuracy",
        value=round(correct / len(should_refuse), 4),
        detail=f"正确拒答 {correct}/{len(should_refuse)} 道不可答题",
        extras={"should_refuse": len(should_refuse), "correct": correct},
    )


def false_refusal_rate(refused_flags: list[bool], answerable_flags: list[bool]) -> MetricResult:
    """误拒率 = 可回答问题中被错误拒答的比例（越低越好）"""
    answerable = [i for i, ok in enumerate(answerable_flags) if ok]
    if not answerable:
        return MetricResult(name="false_refusal_rate", value=0.0, detail="无可答题")

    false_refusals = sum(1 for i in answerable if refused_flags[i])
    return MetricResult(
        name="false_refusal_rate",
        value=round(false_refusals / len(answerable), 4),
        detail=f"误拒 {false_refusals}/{len(answerable)} 道可答题",
        extras={"answerable": len(answerable), "false_refusals": false_refusals},
    )


def citation_validity(valid: int, total: int) -> MetricResult:
    """
    引用编号有效率 = 有效编号数 / 出现过的编号数。

    这是纯正则的零成本指标，能抓到判官抓不到的「编造引用编号」类幻觉
    （判官只看语义，不看编号是否真实存在）。
    """
    if total <= 0:
        return MetricResult(
            name="citation_validity", value=1.0, detail="答案未包含引用标记"
        )
    return MetricResult(
        name="citation_validity",
        value=round(valid / total, 4),
        detail=f"{valid}/{total} 个引用编号在有效范围内",
    )


# ═══════════════════════════════════════════════════════════════
# 数据集级汇总
# ═══════════════════════════════════════════════════════════════

def aggregate(values: list[float]) -> float:
    """求均值（空列表返回 0.0，避免 NaN 传播到报告里）"""
    return round(sum(values) / len(values), 4) if values else 0.0
