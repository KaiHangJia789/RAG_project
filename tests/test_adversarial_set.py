"""
对抗性评测集校验（纯数据校验，零 API 成本）

评测集本身的错误会伪装成「系统能力不足」—— 你会去 debug 检索代码，
而问题其实在题目。这个测试就是防那种情况。
"""
import json
import pathlib
from collections import Counter

import pytest

PATH = pathlib.Path("data/eval/adversarial_eval_set.json")
CORPUS_DIR = pathlib.Path("data/corpus")

pytestmark = pytest.mark.skipif(not PATH.exists(), reason="对抗集尚未生成")

VALID_CATEGORIES = {"premise", "multihop", "out_of_scope", "ambiguous", "injection"}
VALID_BEHAVIORS = {
    "reject_premise", "answer_grounded", "refuse", "clarify_or_cover", "contain_injection",
}


@pytest.fixture(scope="module")
def data() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def items(data) -> list[dict]:
    return data["items"]


class TestStructure:
    def test_exactly_20_questions(self, items):
        """v3 验收要求：20 个对抗性/边界测试问题"""
        assert len(items) == 20

    def test_ids_unique(self, items):
        ids = [i["id"] for i in items]
        assert len(ids) == len(set(ids))

    def test_ids_use_adv_prefix(self, items):
        """adv 前缀便于与常规集的 q*/n* 区分"""
        assert all(i["id"].startswith("adv") for i in items)

    def test_required_fields(self, items):
        for i in items:
            for field in ("id", "category", "expected_behavior", "question",
                          "ground_truth", "answerable"):
                assert field in i, f"{i['id']} 缺字段 {field}"

    def test_categories_valid(self, items):
        for i in items:
            assert i["category"] in VALID_CATEGORIES, \
                f"{i['id']} 类别非法: {i['category']}"

    def test_expected_behaviors_valid(self, items):
        for i in items:
            assert i["expected_behavior"] in VALID_BEHAVIORS, \
                f"{i['id']} 期望行为非法: {i['expected_behavior']}"


class TestCategoryBalance:
    def test_five_categories(self, items):
        cats = {i["category"] for i in items}
        assert cats == VALID_CATEGORIES

    def test_four_per_category(self, items):
        """每类恰好 4 题，保证各类都被充分覆盖"""
        counts = Counter(i["category"] for i in items)
        for cat, n in counts.items():
            assert n == 4, f"类别 {cat} 有 {n} 题，应为 4"


class TestBehaviorConsistency:
    """期望行为必须与类别自洽 —— 不一致说明出题时写错了"""

    def test_refuse_only_for_out_of_scope(self, items):
        for i in items:
            if i["expected_behavior"] == "refuse":
                assert i["category"] == "out_of_scope", \
                    f"{i['id']}: refuse 只应用于 out_of_scope"

    def test_out_of_scope_must_be_unanswerable(self, items):
        for i in items:
            if i["category"] == "out_of_scope":
                assert i["answerable"] is False, \
                    f"{i['id']}: 超范围题必须标为不可回答"

    def test_injection_must_contain(self, items):
        for i in items:
            if i["category"] == "injection":
                assert i["expected_behavior"] == "contain_injection"

    def test_multihop_needs_two_sources(self, items):
        """多跳题必须引用 ≥2 个来源，否则它根本不是多跳"""
        for i in items:
            if i["category"] == "multihop":
                n = len(i.get("reference_sources") or [])
                assert n >= 2, f"{i['id']} 只标了 {n} 个来源，多跳题需要 ≥2"

    def test_premise_needs_answerable_true(self, items):
        """错误前提题是可答的（答案就是「你的前提错了」），不该标为不可答"""
        for i in items:
            if i["category"] == "premise":
                assert i["answerable"] is True, \
                    f"{i['id']}: 前提纠错题应可回答"


class TestReferenceSources:
    def test_referenced_docs_exist(self, items):
        corpus = {p.name for p in CORPUS_DIR.glob("*.md")}
        corpus |= {p.name for p in CORPUS_DIR.glob("*.txt")}
        for i in items:
            for s in i.get("reference_sources") or []:
                assert s["document"] in corpus, \
                    f"{i['id']} 引用了不存在的文档 {s['document']}"

    def test_questions_have_minimum_length(self, items):
        for i in items:
            assert len(i["question"]) >= 5, f"{i['id']} 问题过短"

    def test_ground_truth_is_substantial(self, items):
        """标准答案要写清「正确行为是什么」，不能是敷衍的几个字"""
        for i in items:
            assert len(i["ground_truth"]) >= 30, f"{i['id']} 标准答案过短"

    def test_has_notes(self, items):
        """每题的考察意图要写清楚，否则后人无法理解为什么这样设计"""
        for i in items:
            assert i.get("notes"), f"{i['id']} 缺少考察意图说明"
