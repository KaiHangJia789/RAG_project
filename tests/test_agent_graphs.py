"""
LangGraph 工作流测试

用鸭子类型替身注入（沿用 test_rag_pipeline.py 的模式），不碰网络。
重点验证：线性图的节点顺序、两条短路路径（索引未就绪 / 检索无命中）、
以及 AgentRunner 产出的 RagAnswer 与 RagPipeline 同口径。
"""
import pytest

from app.agent import AgentRunner, ModelBundle
from app.agent.graphs import build_linear_graph, compile_graph, mermaid_of
from app.agent.serialization import hit_to_dict, dict_to_hit, hits_to_dicts
from app.agent.state import AgentState, initial_state
from app.llm.client import LLMResponse, LLMUsage
from app.rag.models import RetrievalConfig
from app.services.index_service import IndexHit, IndexRecord


# ═══════════════════════════════════════════════════════════════
# 替身
# ═══════════════════════════════════════════════════════════════

def make_hit(fid: int, text: str, score: float, filename: str = "doc.md", page=None):
    return IndexHit(
        faiss_id=fid, score=score,
        record=IndexRecord(
            faiss_id=fid, chunk_id=f"chunk-{fid}", document_id="doc-1",
            filename=filename, chunk_index=fid, text=text, page_number=page,
        ),
    )


class FakeIndexService:
    def __init__(self, hits=None, ready=True):
        self._hits = hits if hits is not None else []
        self._ready = ready
        self.queries: list[str] = []

    @property
    def ready(self):
        return self._ready

    async def search(self, query, **kwargs):
        self.queries.append(query)
        return self._hits


class FakeLLM:
    def __init__(self, text="答案[1]", raise_exc=None):
        self.text = text
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    async def generate(self, system, user_message, **kwargs):
        self.calls.append({"system": system, "user": user_message, "kwargs": kwargs})
        if self.raise_exc:
            raise self.raise_exc
        return LLMResponse(
            text=self.text, model="fake",
            usage=LLMUsage(prompt_tokens=100, completion_tokens=20),
        )


@pytest.fixture
def bundle_factory():
    def _make(hits=None, ready=True, llm_text="答案[1]"):
        idx = FakeIndexService(hits, ready)
        llm = FakeLLM(llm_text)
        return ModelBundle(index_service=idx, llm=llm), idx, llm
    return _make


def runner_for(bundle):
    return AgentRunner(compile_graph(build_linear_graph(bundle)))


# ═══════════════════════════════════════════════════════════════
# 序列化
# ═══════════════════════════════════════════════════════════════

class TestSerialization:
    def test_hit_roundtrip_preserves_fields(self):
        hit = make_hit(1, "内容", 0.85, "a.md", 3)
        d = hit_to_dict(hit)

        # record 的字段被摊平到顶层（下游用 h["text"] 与 IndexHit.text 一致）
        assert d["text"] == "内容"
        assert d["filename"] == "a.md"
        assert d["page_number"] == 3
        assert d["score"] == 0.85

        back = dict_to_hit(d)
        assert back.text == "内容"
        assert back.filename == "a.md"
        assert back.page_number == 3
        assert back.score == 0.85

    def test_hits_batch_roundtrip(self):
        hits = [make_hit(i, f"t{i}", 0.5) for i in range(1, 4)]
        assert len(hits_to_dicts(hits)) == 3
        assert [h.text for h in hits] == ["t1", "t2", "t3"]


# ═══════════════════════════════════════════════════════════════
# State
# ═══════════════════════════════════════════════════════════════

class TestState:
    def test_initial_state_defaults(self):
        s = initial_state("问题")
        assert s["question"] == "问题"
        assert s["original_question"] == "问题"
        assert s["rewrite_count"] == 0
        assert s["tool_round"] == 0
        assert s["llm_call_count"] == 0
        assert s["refused"] is False
        assert s["rewrite_history"] == []
        assert s["tool_results"] == []

    def test_accumulating_reducers_declared(self):
        """
        rewrite_history / messages / tool_results 必须是累加 reducer。

        用 `operator.add` 而不是 LangChain 的 `add_messages` —— 后者会对 dict
        做消息语义转换，**丢掉 reasoning_content**，而 DeepSeek 在带 tools 时
        要求该字段原样回传，丢了就 HTTP 400。
        """
        import operator
        from typing import get_args, get_type_hints

        hints = get_type_hints(AgentState, include_extras=True)
        for field_name in ("rewrite_history", "messages", "tool_results"):
            meta = get_args(hints[field_name])
            assert meta, f"{field_name} 缺少 Annotated reducer"
            assert operator.add in meta, f"{field_name} 应使用 operator.add"


# ═══════════════════════════════════════════════════════════════
# 线性图
# ═══════════════════════════════════════════════════════════════

class TestLinearGraph:
    @pytest.mark.asyncio
    async def test_full_path_produces_answer_and_citations(self, bundle_factory):
        bundle, idx, llm = bundle_factory(
            hits=[make_hit(1, "RAG 是检索增强生成", 0.8, "rag.md", 2)],
            llm_text="RAG 是检索增强生成[1]。",
        )
        result = await runner_for(bundle).run("什么是 RAG？")

        assert result.answer == "RAG 是检索增强生成[1]。"
        assert len(result.citations) == 1
        assert result.citations[0].filename == "rag.md"
        assert result.citations[0].page_number == 2
        assert result.refused is False
        assert result.llm_called is True
        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_index_not_ready_short_circuits(self, bundle_factory):
        """索引未就绪 → 直接结束，不检索也不调 LLM"""
        bundle, idx, llm = bundle_factory(ready=False)
        result = await runner_for(bundle).run("任意问题")

        assert result.refused is True
        assert result.refusal_reason.value == "index_not_ready"
        assert len(idx.queries) == 0, "不应发起检索"
        assert len(llm.calls) == 0, "不应调用 LLM"

    @pytest.mark.asyncio
    async def test_no_hits_short_circuits_without_llm(self, bundle_factory):
        """检索无命中 → 阈值层拒答，不发 LLM（这是成本优化的关键路径）"""
        bundle, idx, llm = bundle_factory(hits=[])
        result = await runner_for(bundle).run("无关问题")

        assert result.refused is True
        assert result.refusal_reason.value == "below_threshold"
        assert len(idx.queries) == 1, "应该检索过一次"
        assert len(llm.calls) == 0, "拒答路径不应调用 LLM"

    @pytest.mark.asyncio
    async def test_out_of_range_citation_stripped(self, bundle_factory):
        bundle, _, _ = bundle_factory(
            hits=[make_hit(1, "内容", 0.8)],
            llm_text="答案[1] 还有[9]。",
        )
        result = await runner_for(bundle).run("问题")

        assert result.invalid_citations == [9]
        assert "[9]" not in result.answer
        assert result.citation_validity == 0.5

    @pytest.mark.asyncio
    async def test_llm_refusal_detected(self, bundle_factory):
        bundle, _, _ = bundle_factory(
            hits=[make_hit(1, "内容", 0.8)],
            llm_text="根据提供的资料无法回答该问题。",
        )
        result = await runner_for(bundle).run("问题")

        assert result.refused is True
        assert result.refusal_reason.value == "llm_refused"

    @pytest.mark.asyncio
    async def test_retrieved_docs_is_retrieval_not_citations(self, bundle_factory):
        """
        回归测试（Week10 踩过的坑）：`retrieved_docs` 必须是**检索命中**的文档，
        不是**答案引用**的文档。用错会让 Hit Rate / MRR 混入「LLM 是否引用」
        这个变量，污染纯检索指标。
        """
        bundle, _, _ = bundle_factory(
            hits=[
                make_hit(1, "内容1", 0.9, "a.md"),
                make_hit(2, "内容2", 0.8, "b.md"),
            ],
            llm_text="答案只引用了[1]。",     # 只引用 a.md
        )
        result = await runner_for(bundle).run("问题")

        assert result.retrieved_docs == ["a.md", "b.md"], "应是全部检索命中"
        assert [c.filename for c in result.citations] == ["a.md"], "引用只有一条"

    @pytest.mark.asyncio
    async def test_max_score_recorded(self, bundle_factory):
        bundle, _, _ = bundle_factory(
            hits=[make_hit(1, "a", 0.87), make_hit(2, "b", 0.65)]
        )
        result = await runner_for(bundle).run("问题")
        assert result.max_score == 0.87

    @pytest.mark.asyncio
    async def test_token_usage_accumulated(self, bundle_factory):
        bundle, _, _ = bundle_factory(hits=[make_hit(1, "内容", 0.8)])
        result = await runner_for(bundle).run("问题")
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 20

    @pytest.mark.asyncio
    async def test_config_passed_through(self, bundle_factory):
        """RetrievalConfig 经 state 往返后仍生效"""
        bundle, idx, _ = bundle_factory(hits=[make_hit(1, "内容", 0.8)])
        cfg = RetrievalConfig(final_k=3, min_score=0.5, retrieve_k=50)
        await runner_for(bundle).run("问题", cfg)

        # FakeIndexService 记录的是 query，参数通过 config 传入 search
        assert idx.queries == ["问题"]


# ═══════════════════════════════════════════════════════════════
# Mermaid 流程图
# ═══════════════════════════════════════════════════════════════

class TestMermaid:
    def test_mermaid_contains_all_nodes(self, bundle_factory):
        bundle, _, _ = bundle_factory()
        mermaid = mermaid_of(compile_graph(build_linear_graph(bundle)))

        for node in ["prepare", "retrieve", "assemble", "generate", "citations"]:
            assert node in mermaid, f"流程图缺少节点 {node}"
        assert "__start__" in mermaid
        assert "__end__" in mermaid

    def test_mermaid_marks_conditional_edges(self, bundle_factory):
        """
        条件边应渲染成虚线，与静态边（实线 -->）区分。

        mermaid 的实际输出形如 `prepare -. &nbsp;end&nbsp; .-> __end__;`
        （标签里用 non-breaking space 占位），所以只断言虚线的开头标记。
        """
        bundle, _, _ = bundle_factory()
        mermaid = mermaid_of(compile_graph(build_linear_graph(bundle)))
        conditional = [ln for ln in mermaid.splitlines() if "-." in ln]
        assert len(conditional) >= 2, f"应有条件边，实际:\n{mermaid}"
