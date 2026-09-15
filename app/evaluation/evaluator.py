"""
RAGEvaluator — 单题评估编排

职责：对一次 RAG 问答结果算出全部指标。

设计要点:
  - **判官调用只发生在这里一次**。faithfulness 和 hallucination_rate 复用
    同一批判定结果 —— 若拆成两个独立评估器各自调判官，调用量翻倍且
    两条指标基于不同判定、互不自洽。
  - 负样本（answerable=False）走单独路径：不算忠实度（拒答天然忠实），
    只判断是否正确拒答。
"""
import logging

from pydantic import BaseModel, Field

from app.evaluation.judge import LLMJudge
from app.evaluation.metrics import (
    citation_validity,
    context_precision_ranked,
    faithfulness,
    hallucination_rate,
    hit_rate_at_k,
    mrr,
    source_precision,
)
from app.rag.models import RagAnswer

logger = logging.getLogger("rag_api.eval")


class QuestionMetrics(BaseModel):
    """单题的全部指标"""

    question_id: str = ""
    question: str = ""
    answer: str = ""
    answerable: bool = True
    refused: bool = False
    refusal_reason: str | None = None

    # ── 过程数据（便于失败案例分析）──
    retrieved_docs: list[str] = Field(default_factory=list)
    reference_docs: list[str] = Field(default_factory=list)
    retrieved_count: int = 0
    max_score: float | None = None

    # ── 指标 ──
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_precision: float | None = None
    context_precision_source: float | None = None
    hit_rate: float | None = None
    mrr: float | None = None
    citation_validity: float | None = None

    # ── 判定明细（用于人工复核）──
    claims: list[str] = Field(default_factory=list)
    claim_verdicts: list[bool] = Field(default_factory=list)
    is_refusal_judged: bool = False
    degenerate: bool = False               # 无事实断言的退化答案
    judge_malformed: bool = False          # 判官漏判
    reverse_questions: list[str] = Field(default_factory=list)

    latency_ms: float = 0.0
    llm_called: bool = False
    error: str | None = None


class RAGEvaluator:
    """单题评估器"""

    def __init__(self, judge: LLMJudge | None = None) -> None:
        self.judge = judge or LLMJudge()

    async def evaluate_one(
        self,
        result: RagAnswer,
        *,
        question_id: str = "",
        answerable: bool = True,
        reference_answer: str = "",
        reference_docs: list[str] | None = None,
    ) -> QuestionMetrics:
        """
        评估一次问答结果。

        Args:
            result: RAG 流水线的输出
            answerable: 该题是否可回答（负样本为 False）
            reference_answer: 标注的标准答案（供判官理解问题意图）
            reference_docs: 标注的参考来源文档名
        """
        m = QuestionMetrics(
            question_id=question_id,
            question=result.question,
            answer=result.answer,
            answerable=answerable,
            refused=result.refused,
            refusal_reason=result.refusal_reason.value if result.refusal_reason else None,
            retrieved_docs=result.retrieved_docs or [],
            reference_docs=reference_docs or [],
            retrieved_count=result.retrieved_count,
            max_score=result.max_score,
            latency_ms=round(result.latency_ms, 1),
            llm_called=result.llm_called,
        )

        # 引用有效性（纯正则，零成本，无论是否拒答都算）
        total_cites = len(result.citations) + len(result.invalid_citations)
        m.citation_validity = citation_validity(
            len(result.citations), total_cites
        ).value

        # 基于标注来源的确定性检索指标（零 LLM 成本）
        if reference_docs:
            ref_set = set(reference_docs)
            # 用全部 citations 的文档名（去重后保持顺序）
            docs = list(dict.fromkeys(m.retrieved_docs))
            m.hit_rate = hit_rate_at_k(docs, ref_set).value
            m.mrr = mrr(docs, ref_set).value
            m.context_precision_source = source_precision(docs, ref_set).value

        # ── 负样本：只判断是否正确拒答，不评估忠实度 ──
        if not answerable:
            return m

        # 拒答的可答题：不评估忠实度（无内容可评），但仍记录
        if result.refused:
            return m

        if not result.answer.strip():
            m.error = "答案为空"
            return m

        try:
            await self._llm_metrics(m, result, reference_answer)
        except Exception as e:
            logger.error("判官评估失败 [%s]: %s", question_id, e)
            m.error = f"判官评估失败: {e}"

        return m

    async def _llm_metrics(
        self, m: QuestionMetrics, result: RagAnswer, reference_answer: str
    ) -> None:
        """需要调用判官的那部分指标"""
        # 上下文文本（判官核验断言的依据）
        context = "\n\n".join(c.text for c in result.citations)
        if not context:
            context = "（无检索上下文）"

        # ── Faithfulness：拆解 + 批量核验（2 次调用）──
        claims, is_refusal = await self.judge.decompose_claims(result.answer)
        m.claims = claims
        m.is_refusal_judged = is_refusal

        if is_refusal or not claims:
            m.claim_verdicts = []
            f = faithfulness(0, 0, is_refusal=is_refusal)
            m.faithfulness = f.value
            m.degenerate = bool(f.extras.get("degenerate"))
        else:
            verdicts = await self.judge.verify_claims(claims, context)
            m.claim_verdicts = verdicts
            f = faithfulness(sum(verdicts), len(verdicts))
            m.faithfulness = f.value
            # 判定数对不齐 → 标记格式异常
            m.judge_malformed = len(verdicts) != len(claims)

        # ── Context Precision：批量判定片段相关性（1 次调用）──
        if result.citations:
            rels = await self.judge.judge_context_relevance(
                result.question,
                [c.text for c in result.citations],
                reference_answer,
            )
            m.context_precision = context_precision_ranked(rels).value

        # ── Answer Relevance：反推问题 + 复用已有 embedding 不可得，
        #    这里用判官直评（与反推法交叉验证时另行调用）──
        if result.answer.strip():
            m.answer_relevance = await self.judge.score_relevance_direct(
                result.question, result.answer
            )


# ═══════════════════════════════════════════════════════════════
# 数据集级汇总
# ═══════════════════════════════════════════════════════════════

class EvalReport(BaseModel):
    """数据集级评估报告"""

    config: str = ""
    total: int = 0
    answerable: int = 0
    negative: int = 0

    # 均值（仅统计有值的题目）
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_precision: float = 0.0
    context_precision_source: float = 0.0
    hit_rate: float = 0.0
    mrr: float = 0.0
    citation_validity: float = 0.0

    # 计数型
    hallucination_rate: float = 0.0
    refusal_accuracy: float = 0.0
    false_refusal_rate: float = 0.0

    # 诊断
    degenerate_count: int = 0
    judge_malformed_count: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0
    llm_calls: int = 0

    def to_markdown(self, title: str = "评估报告") -> str:
        lines = [
            f"# {title}",
            "",
            f"配置: `{self.config}`",
            "",
            f"题量: {self.total}（可答 {self.answerable} / 负样本 {self.negative}）",
            "",
            "## 质量指标",
            "",
            "| 指标 | 数值 | 说明 |",
            "|---|---|---|",
            f"| Faithfulness | {self.faithfulness:.4f} | 断言级，被上下文支持的比例 |",
            f"| Answer Relevance | {self.answer_relevance:.4f} | 答案切题程度 |",
            f"| Context Precision | {self.context_precision:.4f} | 检索排序质量（判官版） |",
            f"| Context Precision (source) | {self.context_precision_source:.4f} | 检索排序质量（零成本版） |",
            f"| Hit Rate | {self.hit_rate:.4f} | top_k 命中标注来源的比例 |",
            f"| MRR | {self.mrr:.4f} | 首个命中的倒数排名 |",
            f"| Citation Validity | {self.citation_validity:.4f} | 引用编号有效率 |",
            "",
            "## 幻觉与拒答",
            "",
            "| 指标 | 数值 | 说明 |",
            "|---|---|---|",
            f"| 幻觉率 | {self.hallucination_rate:.4f} | 答案级：含 ≥1 条无支撑断言的比例 |",
            f"| 拒答正确率 | {self.refusal_accuracy:.4f} | 负样本被正确拒答的比例 |",
            f"| 误拒率 | {self.false_refusal_rate:.4f} | 可答题被错误拒答的比例 |",
            "",
            "## 运行诊断",
            "",
            f"- 平均延迟: {self.avg_latency_ms:.0f} ms",
            f"- LLM 调用次数: {self.llm_calls}",
            f"- 退化答案（无事实断言）: {self.degenerate_count}",
            f"- 判官漏判: {self.judge_malformed_count}",
            f"- 评估出错: {self.error_count}",
        ]
        return "\n".join(lines)


def summarize(results: list[QuestionMetrics], config: str = "") -> EvalReport:
    """把逐题结果汇总成报告"""
    from app.evaluation.metrics import (
        aggregate,
        false_refusal_rate,
        hallucination_rate,
        refusal_accuracy,
    )

    total = len(results)
    answerable_flags = [r.answerable for r in results]
    refused_flags = [r.refused for r in results]

    # 幻觉率：只在「可答 + 未拒答 + 有断言」的答案上统计
    evaluable = [
        r for r in results
        if r.answerable and not r.refused and r.claims
    ]
    hallucinated = sum(
        1 for r in evaluable
        if r.claim_verdicts and not all(r.claim_verdicts)
    )

    return EvalReport(
        config=config,
        total=total,
        answerable=sum(1 for f in answerable_flags if f),
        negative=sum(1 for f in answerable_flags if not f),
        faithfulness=aggregate([r.faithfulness for r in results if r.faithfulness is not None]),
        answer_relevance=aggregate([r.answer_relevance for r in results if r.answer_relevance is not None]),
        context_precision=aggregate([r.context_precision for r in results if r.context_precision is not None]),
        context_precision_source=aggregate([r.context_precision_source for r in results if r.context_precision_source is not None]),
        hit_rate=aggregate([r.hit_rate for r in results if r.hit_rate is not None]),
        mrr=aggregate([r.mrr for r in results if r.mrr is not None]),
        citation_validity=aggregate([r.citation_validity for r in results if r.citation_validity is not None]),
        hallucination_rate=hallucination_rate(len(evaluable), hallucinated).value,
        refusal_accuracy=refusal_accuracy(refused_flags, answerable_flags).value,
        false_refusal_rate=false_refusal_rate(refused_flags, answerable_flags).value,
        degenerate_count=sum(1 for r in results if r.degenerate),
        judge_malformed_count=sum(1 for r in results if r.judge_malformed),
        error_count=sum(1 for r in results if r.error),
        avg_latency_ms=round(
            sum(r.latency_ms for r in results) / total, 1
        ) if total else 0.0,
        llm_calls=sum(1 for r in results if r.llm_called),
    )
