"""
评测集质量校验

评测集错了，所有指标都失去意义，而且症状会伪装成「检索有 bug」——
你会去 debug 代码，而问题其实在内容。这个测试就是防止那种情况。
"""
import json
import pathlib

import pytest

EVALSET = pathlib.Path("data/eval/rag_eval_set.json")
CORPUS_DIR = pathlib.Path("data/corpus")

pytestmark = pytest.mark.skipif(
    not EVALSET.exists(), reason="评测集尚未生成"
)


@pytest.fixture(scope="module")
def data() -> dict:
    return json.loads(EVALSET.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def items(data) -> list[dict]:
    return data["items"]


class TestEvalSetStructure:
    def test_has_at_least_50_questions(self, items):
        """v3 验收要求：50 个测试问题"""
        assert len(items) >= 50, f"只有 {len(items)} 题，需 ≥50"

    def test_ids_unique(self, items):
        ids = [i["id"] for i in items]
        assert len(ids) == len(set(ids)), "存在重复 ID"

    def test_required_fields_present(self, items):
        """结构化评测集_v1 的三列：问题 / 标准答案 / 参考来源"""
        for i in items:
            assert i.get("question", "").strip(), f"{i['id']} 缺问题"
            assert i.get("ground_truth", "").strip(), f"{i['id']} 缺标准答案"
            assert "reference_sources" in i, f"{i['id']} 缺参考来源字段"

    def test_answerable_flag_is_boolean(self, items):
        for i in items:
            assert isinstance(i.get("answerable"), bool), f"{i['id']} answerable 非布尔"

    def test_has_negative_samples(self, items):
        """必须包含负样本，否则无法评估拒答能力"""
        neg = [i for i in items if not i["answerable"]]
        assert len(neg) >= 5, f"负样本只有 {len(neg)} 道，需 ≥5"

    def test_negative_ratio_reasonable(self, items):
        """负样本占比应在 10%-30% 之间"""
        ratio = sum(1 for i in items if not i["answerable"]) / len(items)
        assert 0.10 <= ratio <= 0.30, f"负样本占比 {ratio:.0%} 不合理"


class TestReferenceSources:
    def test_all_referenced_documents_exist(self, items):
        """
        每题标注的来源文档必须真实存在于语料目录。

        这条错了就是致命的：问题问的内容在语料里根本不存在，
        检索必然失败，而你会以为是检索代码有 bug。
        """
        corpus = {p.name for p in CORPUS_DIR.glob("*.md")}
        corpus |= {p.name for p in CORPUS_DIR.glob("*.txt")}
        assert corpus, "语料目录为空"

        missing: list[tuple[str, str]] = []
        for i in items:
            for s in i.get("reference_sources", []):
                if s["document"] not in corpus:
                    missing.append((i["id"], s["document"]))
        assert not missing, f"标注的来源文档不存在: {missing}"

    def test_positive_samples_have_sources(self, items):
        """可回答的题必须有参考来源（否则无法算 hit_rate / MRR）"""
        for i in items:
            if i["answerable"]:
                assert i.get("reference_sources"), f"{i['id']} 是可答题但没标注来源"

    def test_negative_samples_have_no_sources(self, items):
        """负样本不该有参考来源（语料里本来就没有答案）"""
        for i in items:
            if not i["answerable"]:
                assert not i.get("reference_sources"), f"{i['id']} 是负样本却标了来源"


class TestCorpusCoverage:
    def test_every_corpus_doc_is_referenced(self, items):
        """每篇语料都应被至少一道题引用 —— 否则那篇文档没被评测覆盖"""
        corpus = {p.name for p in CORPUS_DIR.glob("*.md")}
        referenced = {
            s["document"] for i in items for s in i.get("reference_sources", [])
        }
        uncovered = corpus - referenced
        assert not uncovered, f"以下语料没有任何题目引用: {uncovered}"

    def test_corpus_has_enough_content(self):
        """语料量级要撑得起 50 题 —— 太少会导致题目高度重复、无区分度"""
        files = list(CORPUS_DIR.glob("*.md"))
        total = sum(f.stat().st_size for f in files)
        assert total >= 30_000, f"语料仅 {total} 字节，不足以支撑 50 题评测"

    def test_questions_distributed_across_topics(self, items):
        """题目不能集中在单一主题，否则对比实验没区分度"""
        topics = {i.get("topic") for i in items}
        assert len(topics) >= 5, f"主题只有 {topics}"


class TestQuestionQuality:
    def test_questions_not_duplicated(self, items):
        qs = [i["question"].strip() for i in items]
        assert len(qs) == len(set(qs)), "存在完全重复的问题"

    def test_questions_have_minimum_length(self, items):
        for i in items:
            assert len(i["question"]) >= 6, f"{i['id']} 问题过短: {i['question']}"

    def test_ground_truth_substantial(self, items):
        """标准答案不能是敷衍的一两个字"""
        for i in items:
            if i["answerable"]:
                assert len(i["ground_truth"]) >= 20, f"{i['id']} 标准答案过短"
