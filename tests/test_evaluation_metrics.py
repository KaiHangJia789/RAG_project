"""
评估指标测试 — 全部用**手算值**校验

这些指标是整个 Week10 结论的基础。公式实现错了，所有对比数据都是错的，
而且错得很隐蔽（数字看起来都合理）。所以每个指标都用可手算的例子锁死。
"""
import pytest

from app.evaluation.metrics import (
    aggregate,
    citation_validity,
    context_precision_ranked,
    faithfulness,
    false_refusal_rate,
    hallucination_rate,
    hit_rate_at_k,
    mrr,
    refusal_accuracy,
    source_precision,
)


class TestContextPrecision:
    """rank-weighted average precision"""

    def test_hand_calculated_example(self):
        """[1,0,1,0,0]：p@1=1, p@3=2/3 → (1×1 + 1×0.667)/2 = 0.8333"""
        assert context_precision_ranked([1, 0, 1, 0, 0]).value == 0.8333

    def test_all_relevant_is_one(self):
        assert context_precision_ranked([1, 1, 1, 1, 1]).value == 1.0

    def test_none_relevant_is_zero(self):
        assert context_precision_ranked([0, 0, 0, 0, 0]).value == 0.0

    def test_relevant_at_last_position(self):
        """只有一个相关且在末位：p@5 = 1/5 = 0.2"""
        assert context_precision_ranked([0, 0, 0, 0, 1]).value == 0.2

    def test_relevant_at_first_position(self):
        """唯一相关在首位：p@1=1 → 1/1 = 1.0"""
        assert context_precision_ranked([1, 0, 0, 0, 0]).value == 1.0

    def test_empty_returns_zero_not_nan(self):
        assert context_precision_ranked([]).value == 0.0

    def test_accepts_booleans(self):
        assert context_precision_ranked([True, False, True]) == context_precision_ranked([1, 0, 1])

    def test_single_relevant(self):
        """[0,1]: p@2 = 1/2 → 0.5/1 = 0.5"""
        assert context_precision_ranked([0, 1]).value == 0.5


class TestFaithfulness:
    def test_basic_ratio(self):
        assert faithfulness(3, 4).value == 0.75

    def test_all_supported(self):
        assert faithfulness(5, 5).value == 1.0

    def test_none_supported(self):
        assert faithfulness(0, 3).value == 0.0

    def test_refusal_is_fully_faithful(self):
        """拒答不含事实断言，天然忠实，不该被惩罚"""
        r = faithfulness(0, 0, is_refusal=True)
        assert r.value == 1.0
        assert "拒答" in r.detail

    def test_zero_claims_marked_degenerate(self):
        """零断言但非拒答 → 记 1.0 并标记退化，单独统计"""
        r = faithfulness(0, 0)
        assert r.value == 1.0
        assert r.extras.get("degenerate") is True

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            faithfulness(5, 3)
        with pytest.raises(ValueError):
            faithfulness(-1, 3)


class TestHallucinationRate:
    def test_basic(self):
        assert hallucination_rate(10, 3).value == 0.3

    def test_no_evaluable_answers(self):
        assert hallucination_rate(0, 0).value == 0.0

    def test_all_hallucinated(self):
        assert hallucination_rate(4, 4).value == 1.0

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            hallucination_rate(3, 5)


class TestRetrievalMetrics:
    REFS = {"01_rag_basics.md", "02_embedding.md"}

    def test_hit_rate_hit(self):
        assert hit_rate_at_k(["01_rag_basics.md", "x.md"], self.REFS).value == 1.0

    def test_hit_rate_miss(self):
        assert hit_rate_at_k(["x.md", "y.md"], self.REFS).value == 0.0

    def test_hit_rate_no_reference_is_zero(self):
        assert hit_rate_at_k(["x.md"], set()).value == 0.0

    def test_mrr_first_position(self):
        assert mrr(["01_rag_basics.md", "x.md"], self.REFS).value == 1.0

    def test_mrr_second_position(self):
        assert mrr(["x.md", "02_embedding.md"], self.REFS).value == 0.5

    def test_mrr_third_position(self):
        assert mrr(["x.md", "y.md", "01_rag_basics.md"], self.REFS).value == 0.3333

    def test_mrr_no_hit(self):
        assert mrr(["x.md", "y.md"], self.REFS).value == 0.0

    def test_source_precision(self):
        """5 条里 1 条来自标注来源 → 0.2"""
        assert source_precision(
            ["01_rag_basics.md", "x.md", "y.md", "z.md", "w.md"], self.REFS
        ).value == 0.2

    def test_source_precision_empty(self):
        assert source_precision([], self.REFS).value == 0.0


class TestRefusalMetrics:
    def test_refusal_accuracy_all_correct(self):
        """3 题：前两题可答未拒，第 3 题不可答且拒了 → 1/1 = 1.0"""
        assert refusal_accuracy([False, False, True], [True, True, False]).value == 1.0

    def test_refusal_accuracy_missed(self):
        """负样本没拒 → 0.0"""
        assert refusal_accuracy([False, False, False], [True, True, False]).value == 0.0

    def test_refusal_accuracy_no_negatives(self):
        assert refusal_accuracy([False, False], [True, True]).value == 0.0

    def test_false_refusal_rate(self):
        """2 道可答题里 1 道被误拒 → 0.5"""
        assert false_refusal_rate([True, False, True], [True, True, False]).value == 0.5

    def test_false_refusal_rate_none(self):
        assert false_refusal_rate([False, False], [True, True]).value == 0.0


class TestCitationValidity:
    def test_partial(self):
        assert citation_validity(3, 4).value == 0.75

    def test_no_citations_is_perfect(self):
        """没引用不算无效 —— 由别的指标衡量「该引没引」"""
        assert citation_validity(0, 0).value == 1.0

    def test_all_invalid(self):
        assert citation_validity(0, 3).value == 0.0


class TestAggregate:
    def test_mean(self):
        assert aggregate([1.0, 0.5, 0.0]) == 0.5

    def test_empty_is_zero_not_nan(self):
        assert aggregate([]) == 0.0
