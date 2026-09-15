"""
判官测试 — JSON 容错解析、批量判定对齐、缓存

不调真实 API，用鸭子类型 FakeLLM 返回预录 JSON。
重点覆盖判官不稳定的三种表现：格式漂移、乱序返回、漏判。
"""
import json
import pathlib

import pytest

from app.evaluation.judge import JudgeCache, LLMJudge, parse_llm_json
from app.evaluation.prompts import JUDGE_PROMPTS, PROMPT_VERSION, get_judge_prompt
from app.llm.client import LLMResponse


class FakeLLM:
    """返回预录文本的假 LLM，记录调用参数"""

    def __init__(self, text: str = "{}", raise_on_call: bool = False):
        self.text = text
        self.raise_on_call = raise_on_call
        self.calls: list[dict] = []

    async def generate(self, system, user_message, **kwargs):
        self.calls.append({"system": system, "user": user_message, "kwargs": kwargs})
        if self.raise_on_call:
            raise RuntimeError("模拟调用失败")
        return LLMResponse(text=self.text, model="fake")


class FlakyLLM:
    """前 n 次返回坏 JSON，之后返回好 JSON（测重试）"""

    def __init__(self, good: str, bad_count: int = 1):
        self.good = good
        self.bad_count = bad_count
        self.n = 0

    async def generate(self, system, user_message, **kwargs):
        self.n += 1
        text = "这不是 JSON" if self.n <= self.bad_count else self.good
        return LLMResponse(text=text, model="fake")


@pytest.fixture
def cache(tmp_path) -> JudgeCache:
    return JudgeCache(tmp_path / "cache.jsonl")


# ═══════════════════════════════════════════════════════════════
# JSON 容错解析
# ═══════════════════════════════════════════════════════════════

class TestParseLlmJson:
    def test_plain_json(self):
        assert parse_llm_json('{"a": 1}') == {"a": 1}

    def test_markdown_fence(self):
        assert parse_llm_json('```json\n{"b": 2}\n```') == {"b": 2}

    def test_fence_without_language(self):
        assert parse_llm_json('```\n{"c": 3}\n```') == {"c": 3}

    def test_surrounding_prose(self):
        """模型常会在 JSON 前后加解释性文字"""
        assert parse_llm_json('好的，我的判断是：{"d": 4} 以上。') == {"d": 4}

    def test_nested_braces(self):
        assert parse_llm_json('{"e": {"f": 5}}') == {"e": {"f": 5}}

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_llm_json("")
        with pytest.raises(ValueError):
            parse_llm_json("   ")

    def test_non_json_raises(self):
        with pytest.raises(ValueError):
            parse_llm_json("完全不是 JSON 的内容")


# ═══════════════════════════════════════════════════════════════
# 判官 Prompt 注册表
# ═══════════════════════════════════════════════════════════════

class TestJudgePrompts:
    def test_registry_has_all_five(self):
        assert len(JUDGE_PROMPTS) == 5
        for name in [
            "faithfulness_decompose", "faithfulness_verify",
            "relevance_reverse_questions", "context_relevance_judge",
            "relevance_direct_score",
        ]:
            assert name in JUDGE_PROMPTS

    def test_judge_prompts_isolated_from_business_registry(self):
        """
        判官 prompt 绝不能进 PROMPT_REGISTRY。

        tests/test_prompts.py::test_all_prompts_renderable 会用固定的
        {input, context, language} 渲染注册表里的每个模板；判官模板需要
        {answer}/{claims}/{chunks}，混进去会让那个测试 KeyError。
        """
        from app.llm.prompts import PROMPT_REGISTRY

        overlap = set(JUDGE_PROMPTS) & set(PROMPT_REGISTRY)
        assert not overlap, f"判官模板混入了业务注册表: {overlap}"

    def test_all_judge_prompts_renderable(self):
        """判官模板自己必须都能渲染（用各自的真实参数）"""
        args = {
            "faithfulness_decompose": {"answer": "答案"},
            "faithfulness_verify": {"context": "上下文", "claims": "1. 断言"},
            "relevance_reverse_questions": {"answer": "答案"},
            "context_relevance_judge": {
                "question": "问题", "reference_answer": "标准答案", "chunks": "[1] 片段"
            },
            "relevance_direct_score": {"question": "问题", "answer": "答案"},
        }
        for name, tpl in JUDGE_PROMPTS.items():
            system, user = tpl.render(**args[name])
            assert system and user

    def test_unknown_prompt_raises(self):
        with pytest.raises(ValueError):
            get_judge_prompt("不存在的模板")

    def test_prompts_mention_json(self):
        """json_mode 要求 prompt 里出现字面量 json，否则部分提供商会报错"""
        for tpl in JUDGE_PROMPTS.values():
            assert "json" in tpl.system.lower(), f"{tpl.name} 缺少 json 字面量"

    def test_prompt_version_exists(self):
        assert PROMPT_VERSION


# ═══════════════════════════════════════════════════════════════
# 判官能力
# ═══════════════════════════════════════════════════════════════

class TestDecomposeClaims:
    @pytest.mark.asyncio
    async def test_normal(self, cache):
        llm = FakeLLM('{"claims": ["断言A", "断言B"], "is_refusal": false}')
        j = LLMJudge(llm=llm, cache=cache)
        claims, is_refusal = await j.decompose_claims("答案")
        assert claims == ["断言A", "断言B"]
        assert is_refusal is False

    @pytest.mark.asyncio
    async def test_refusal(self, cache):
        j = LLMJudge(llm=FakeLLM('{"claims": [], "is_refusal": true}'), cache=cache)
        claims, is_refusal = await j.decompose_claims("无法回答")
        assert claims == []
        assert is_refusal is True

    @pytest.mark.asyncio
    async def test_filters_empty_and_null_claims(self, cache):
        """模型偶尔会塞 null 或空串进来"""
        j = LLMJudge(
            llm=FakeLLM('{"claims": ["有效", "", null, "  "], "is_refusal": false}'),
            cache=cache,
        )
        claims, _ = await j.decompose_claims("答案")
        assert claims == ["有效"]

    @pytest.mark.asyncio
    async def test_judge_params_are_deterministic(self, cache):
        """判官必须关思考 + 温度 0 —— 稳定性与速度的关键"""
        llm = FakeLLM('{"claims": [], "is_refusal": false}')
        j = LLMJudge(llm=llm, cache=cache)
        await j.decompose_claims("答案")

        kwargs = llm.calls[0]["kwargs"]
        assert kwargs["thinking_enabled"] is False
        assert kwargs["temperature"] == 0.0
        assert kwargs["json_mode"] is True


class TestVerifyClaims:
    @pytest.mark.asyncio
    async def test_ordered_response(self, cache):
        j = LLMJudge(
            llm=FakeLLM('{"verdicts": [{"id":1,"supported":true},{"id":2,"supported":false}]}'),
            cache=cache,
        )
        assert await j.verify_claims(["A", "B"], "ctx") == [True, False]

    @pytest.mark.asyncio
    async def test_out_of_order_response(self, cache):
        """模型可能乱序返回，必须按 id 对齐"""
        j = LLMJudge(
            llm=FakeLLM('{"verdicts": [{"id":3,"supported":true},{"id":1,"supported":true},{"id":2,"supported":false}]}'),
            cache=cache,
        )
        assert await j.verify_claims(["A", "B", "C"], "ctx") == [True, False, True]

    @pytest.mark.asyncio
    async def test_missing_verdict_treated_as_unsupported(self, cache):
        """漏判按「不支持」处理（宁可低估），不静默补齐"""
        j = LLMJudge(
            llm=FakeLLM('{"verdicts": [{"id":1,"supported":true}]}'),
            cache=cache,
        )
        assert await j.verify_claims(["A", "B", "C"], "ctx") == [True, False, False]

    @pytest.mark.asyncio
    async def test_empty_claims_no_call(self, cache):
        llm = FakeLLM("{}")
        j = LLMJudge(llm=llm, cache=cache)
        assert await j.verify_claims([], "ctx") == []
        assert len(llm.calls) == 0, "无断言时不应调用模型"


class TestContextRelevance:
    @pytest.mark.asyncio
    async def test_ordered(self, cache):
        j = LLMJudge(
            llm=FakeLLM('{"judgments":[{"id":1,"relevant":true},{"id":2,"relevant":false}]}'),
            cache=cache,
        )
        assert await j.judge_context_relevance("q", ["c1", "c2"], "") == [True, False]

    @pytest.mark.asyncio
    async def test_missing_treated_as_irrelevant(self, cache):
        j = LLMJudge(llm=FakeLLM('{"judgments":[]}'), cache=cache)
        assert await j.judge_context_relevance("q", ["c1", "c2"], "") == [False, False]

    @pytest.mark.asyncio
    async def test_empty_chunks_no_call(self, cache):
        llm = FakeLLM("{}")
        j = LLMJudge(llm=llm, cache=cache)
        assert await j.judge_context_relevance("q", [], "") == []
        assert len(llm.calls) == 0


class TestRelevanceScore:
    @pytest.mark.asyncio
    async def test_clamps_above_one(self, cache):
        j = LLMJudge(llm=FakeLLM('{"score": 1.5}'), cache=cache)
        assert await j.score_relevance_direct("q", "a") == 1.0

    @pytest.mark.asyncio
    async def test_clamps_below_zero(self, cache):
        # 答案必须与上一条不同 —— 缓存键基于渲染后的 prompt，
        # 参数完全相同会命中缓存而拿不到新结果
        j = LLMJudge(llm=FakeLLM('{"score": -0.3}'), cache=cache)
        assert await j.score_relevance_direct("q", "另一个答案") == 0.0

    @pytest.mark.asyncio
    async def test_invalid_score_returns_zero(self, cache):
        j = LLMJudge(llm=FakeLLM('{"score": "很高"}'), cache=cache)
        assert await j.score_relevance_direct("q", "a") == 0.0


class TestRetryAndCache:
    @pytest.mark.asyncio
    async def test_retries_once_on_bad_json(self, cache):
        """格式漂移应重试一次并成功"""
        llm = FlakyLLM('{"claims": ["A"], "is_refusal": false}', bad_count=1)
        j = LLMJudge(llm=llm, cache=cache)
        claims, _ = await j.decompose_claims("答案")
        assert claims == ["A"]
        assert llm.n == 2

    @pytest.mark.asyncio
    async def test_gives_up_after_two_failures(self, cache):
        llm = FlakyLLM('{"claims": []}', bad_count=5)
        j = LLMJudge(llm=llm, cache=cache)
        with pytest.raises(RuntimeError, match="判官调用最终失败"):
            await j.decompose_claims("答案")
        assert llm.n == 2

    @pytest.mark.asyncio
    async def test_cache_hit_avoids_call(self, cache):
        llm = FakeLLM('{"claims": ["A"], "is_refusal": false}')
        j = LLMJudge(llm=llm, cache=cache)

        await j.decompose_claims("相同答案")
        n_after_first = len(llm.calls)
        await j.decompose_claims("相同答案")

        assert len(llm.calls) == n_after_first, "第二次应命中缓存不发调用"
        assert cache.stats["hits"] >= 1

    def test_cache_persists_to_disk(self, tmp_path):
        path = tmp_path / "c.jsonl"
        c1 = JudgeCache(path)
        c1.put("k1", {"v": 1})

        c2 = JudgeCache(path)
        assert c2.get("k1") == {"v": 1}

    def test_cache_key_includes_prompt_version(self, monkeypatch):
        """
        换 prompt 版本必须让旧缓存失效 —— 否则新口径的分数会混入旧判定结果。

        补丁要打在 judge 模块上：judge.py 用的是
        `from ...prompts import PROMPT_VERSION`，绑定的是值，改 prompts 模块无效。
        """
        import app.evaluation.judge as judge_mod

        k1 = JudgeCache.make_key("m", "p", "payload")
        monkeypatch.setattr(judge_mod, "PROMPT_VERSION", "judge-v999")
        k2 = JudgeCache.make_key("m", "p", "payload")

        assert k1 != k2

    def test_cache_key_includes_model_and_prompt_name(self):
        base = JudgeCache.make_key("m", "p", "x")
        assert base != JudgeCache.make_key("m2", "p", "x")
        assert base != JudgeCache.make_key("m", "p2", "x")
        assert base != JudgeCache.make_key("m", "p", "y")
