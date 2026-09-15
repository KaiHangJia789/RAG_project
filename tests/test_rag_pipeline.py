"""
RAG 问答链路测试

用鸭子类型替身注入（照 test_experiments.py 的模式），不依赖真实 DB/网络。
重点覆盖三条容易出错的路径：阈值拒答、引用越界、索引未就绪。
"""
import pytest

from app.llm.client import LLMResponse, LLMUsage
from app.rag.citations import (
    build_context,
    extract_citation_indices,
    has_citation,
    is_refusal_lexical,
    strip_invalid_citations,
    validate_citations,
)
from app.rag.models import RefusalReason, RetrievalConfig
from app.rag.pipeline import RagPipeline
from app.services.index_service import IndexHit, IndexRecord


# ═══════════════════════════════════════════════════════════════
# 替身
# ═══════════════════════════════════════════════════════════════

def make_hit(fid: int, text: str, score: float, filename: str = "doc.md", page: int | None = 1):
    return IndexHit(
        faiss_id=fid, score=score,
        record=IndexRecord(
            faiss_id=fid, chunk_id=f"chunk-{fid}", document_id="doc-1",
            filename=filename, chunk_index=fid, text=text, page_number=page,
        ),
    )


class FakeIndexService:
    """鸭子类型替身：只实现 pipeline 用到的方法"""

    def __init__(self, hits=None, ready=True):
        self._hits = hits or []
        self._ready = ready
        self.search_calls = []

    @property
    def ready(self):
        return self._ready

    @property
    def strategy(self):
        return "fake"

    async def search(self, query, **kwargs):
        self.search_calls.append({"query": query, **kwargs})
        return self._hits

    def get_by_chunk_id(self, chunk_id):
        for h in self._hits:
            if h.record.chunk_id == chunk_id:
                return h.record
        return None

    def stats(self):
        return {"strategy": "fake", "dim": 8, "vectors": len(self._hits),
                "records": len(self._hits), "documents": 1, "pages": 1,
                "index_path": "x", "exists": True}


class FakeLLM:
    """记录调用参数的假 LLM"""

    def __init__(self, text="答案[1]", should_raise=False):
        self.text = text
        self.should_raise = should_raise
        self.calls = []

    async def generate(self, system, user_message, **kwargs):
        self.calls.append({"system": system, "user": user_message, "kwargs": kwargs})
        if self.should_raise:
            raise RuntimeError("LLM 挂了")
        return LLMResponse(
            text=self.text, model="fake",
            usage=LLMUsage(prompt_tokens=100, completion_tokens=20),
        )


# ═══════════════════════════════════════════════════════════════
# 引用处理（纯函数）
# ═══════════════════════════════════════════════════════════════

class TestCitationParsing:
    def test_extract_dedupes_and_keeps_order(self):
        assert extract_citation_indices("A[1] B[2] C[1]") == [1, 2]

    def test_extract_fullwidth_brackets(self):
        assert extract_citation_indices("答案［3］") == [3]

    def test_extract_none(self):
        assert extract_citation_indices("没有引用") == []

    def test_validate_all_valid(self):
        assert validate_citations("见[1][2]", 3) == (1.0, [])

    def test_validate_flags_out_of_range(self):
        validity, invalid = validate_citations("见[1][9]", 3)
        assert validity == 0.5
        assert invalid == [9]

    def test_validate_no_citations_is_perfect(self):
        """没引用不算无效 —— 由别的指标衡量「该引没引」"""
        assert validate_citations("没有引用", 3) == (1.0, [])

    def test_strip_removes_out_of_range_only(self):
        assert strip_invalid_citations("见[1][9]与[2]", 3) == "见[1]与[2]"

    def test_strip_zero_index(self):
        assert strip_invalid_citations("见[0]", 3) == "见"

    def test_has_citation(self):
        assert has_citation("见 [1]") is True
        assert has_citation("见 来源") is True
        assert has_citation("普通文字") is False


class TestRefusalDetection:
    def test_detects_common_refusals(self):
        for text in [
            "根据提供的资料无法回答该问题。",
            "上下文中没有相关信息。",
            "我无法确定。",
            "资料中没有提及。",
        ]:
            assert is_refusal_lexical(text) is True, text

    def test_empty_is_refusal(self):
        assert is_refusal_lexical("") is True

    def test_normal_answer_is_not_refusal(self):
        assert is_refusal_lexical("RAG 是检索增强生成技术。") is False

    def test_long_answer_with_uncertainty_word_is_not_refusal(self):
        """长答案里出现「不知道」多半是在陈述别的内容，不该误判为拒答"""
        long_text = "在讨论不确定性时，模型可能会说不知道。" * 10
        assert is_refusal_lexical(long_text) is False


class TestBuildContext:
    def test_numbers_and_metadata(self):
        ctx = build_context([
            make_hit(1, "内容A", 0.9, "a.md", 3),
            make_hit(2, "内容B", 0.8, "b.md", None),
        ])
        assert "[1] 来源：a.md · 第 3 页" in ctx
        assert "[2] 来源：b.md" in ctx
        assert "第 3 页" in ctx

    def test_respects_max_chars(self):
        ctx = build_context([make_hit(1, "很长的内容" * 500, 0.9)], max_chars=100)
        assert ctx == ""

    def test_empty_hits(self):
        assert build_context([]) == ""


# ═══════════════════════════════════════════════════════════════
# 流水线
# ═══════════════════════════════════════════════════════════════

class TestRagPipeline:
    @pytest.mark.asyncio
    async def test_normal_answer_with_citations(self):
        idx = FakeIndexService([make_hit(1, "RAG 是检索增强生成", 0.8, "rag.md", 2)])
        llm = FakeLLM("RAG 是检索增强生成[1]。")
        p = RagPipeline(index_service=idx, llm=llm)

        r = await p.answer("什么是 RAG？")

        assert r.answer == "RAG 是检索增强生成[1]。"
        assert r.refused is False
        assert len(r.citations) == 1
        assert r.citations[0].filename == "rag.md"
        assert r.citations[0].page_number == 2
        assert r.citations[0].index == 1
        assert r.llm_called is True
        assert r.prompt_tokens == 100

    @pytest.mark.asyncio
    async def test_no_hits_refuses_without_calling_llm(self):
        """阈值层拒答必须跳过 LLM —— 这是成本与延迟优化的关键"""
        idx = FakeIndexService([])
        llm = FakeLLM()
        p = RagPipeline(index_service=idx, llm=llm)

        r = await p.answer("无关问题")

        assert r.refused is True
        assert r.refusal_reason == RefusalReason.BELOW_THRESHOLD
        assert r.llm_called is False
        assert len(llm.calls) == 0, "拒答路径不应调用 LLM"

    @pytest.mark.asyncio
    async def test_index_not_ready(self):
        p = RagPipeline(index_service=FakeIndexService(ready=False), llm=FakeLLM())
        r = await p.answer("任意问题")
        assert r.refused is True
        assert r.refusal_reason == RefusalReason.INDEX_NOT_READY

    @pytest.mark.asyncio
    async def test_out_of_range_citation_flagged_and_stripped(self):
        idx = FakeIndexService([make_hit(1, "内容", 0.8)])
        llm = FakeLLM("答案[1] 还有[9]。")
        p = RagPipeline(index_service=idx, llm=llm)

        r = await p.answer("问题")

        assert r.invalid_citations == [9]
        assert r.citation_validity == 0.5
        assert "[9]" not in r.answer, "越界引用必须从答案中剔除"
        assert "[1]" in r.answer

    @pytest.mark.asyncio
    async def test_llm_refusal_detected(self):
        idx = FakeIndexService([make_hit(1, "内容", 0.8)])
        llm = FakeLLM("根据提供的资料无法回答该问题。")
        p = RagPipeline(index_service=idx, llm=llm)

        r = await p.answer("问题")

        assert r.refused is True
        assert r.refusal_reason == RefusalReason.LLM_REFUSED
        assert r.llm_called is True
        assert r.citations == []

    @pytest.mark.asyncio
    async def test_config_passed_to_search(self):
        idx = FakeIndexService([make_hit(1, "内容", 0.8)])
        p = RagPipeline(index_service=idx, llm=FakeLLM())

        cfg = RetrievalConfig(final_k=3, min_score=0.5, retrieve_k=50, strategy="sentence")
        await p.answer("问题", cfg)

        call = idx.search_calls[0]
        assert call["final_k"] == 3
        assert call["min_score"] == 0.5
        assert call["retrieve_k"] == 50

    @pytest.mark.asyncio
    async def test_unknown_prompt_raises(self):
        idx = FakeIndexService([make_hit(1, "内容", 0.8)])
        p = RagPipeline(index_service=idx, llm=FakeLLM())

        with pytest.raises(ValueError, match="未知 prompt"):
            await p.answer("问题", RetrievalConfig(prompt_name="不存在的模板"))

    @pytest.mark.asyncio
    async def test_citations_only_include_referenced_hits(self):
        """检索到 3 条但答案只引用了 [2] → 只返回那一条引用"""
        idx = FakeIndexService([
            make_hit(1, "内容1", 0.9, "a.md"),
            make_hit(2, "内容2", 0.8, "b.md"),
            make_hit(3, "内容3", 0.7, "c.md"),
        ])
        llm = FakeLLM("答案引用了[2]。")
        p = RagPipeline(index_service=idx, llm=llm)

        r = await p.answer("问题")

        assert len(r.citations) == 1
        assert r.citations[0].index == 2
        assert r.citations[0].filename == "b.md"
        assert r.retrieved_count == 3

    @pytest.mark.asyncio
    async def test_max_score_recorded(self):
        """记录最高相似度 —— 这是阈值标定的关键信号"""
        idx = FakeIndexService([
            make_hit(1, "内容1", 0.87),
            make_hit(2, "内容2", 0.65),
        ])
        p = RagPipeline(index_service=idx, llm=FakeLLM())
        r = await p.answer("问题")
        assert r.max_score == 0.87
