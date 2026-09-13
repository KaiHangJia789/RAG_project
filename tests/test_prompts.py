"""
Prompt 模板测试
"""
import pytest

from app.llm.prompts import PROMPT_REGISTRY, PromptTemplate


class TestPromptTemplate:
    def test_render(self):
        tpl = PromptTemplate(
            name="test",
            description="测试",
            system="系统提示",
            user_template="用户输入: {input}",
        )
        system, user = tpl.render(input="hello")
        assert system == "系统提示"
        assert user == "用户输入: hello"

    def test_render_multiple_placeholders(self):
        tpl = PromptTemplate(
            name="test",
            description="",
            system="",
            user_template="{a} + {b}",
        )
        _, user = tpl.render(a="1", b="2")
        assert user == "1 + 2"


class TestPromptRegistry:
    def test_registry_has_at_least_10_prompts(self):
        """验收要求：Prompt 模板库 v1 ≥10 个场景"""
        assert len(PROMPT_REGISTRY) >= 10

    def test_required_prompt_names(self):
        required = [
            "baseline", "structured", "concise", "json_output",
            "chain_of_thought", "few_shot",
            "query_rewrite", "rag_context", "refuse", "code_gen",
        ]
        for name in required:
            assert name in PROMPT_REGISTRY, f"缺少 prompt: {name}"

    def test_all_prompts_renderable(self):
        """每个模板都能用完整参数渲染"""
        # 各模板所需的完整参数
        full_args = {
            "input": "测试输入",
            "context": "上下文内容",
            "language": "Python",
        }
        for name, tpl in PROMPT_REGISTRY.items():
            system, user = tpl.render(**full_args)
            assert isinstance(system, str)
            assert isinstance(user, str)
            assert len(user) > 0

    def test_query_rewrite_renders(self):
        tpl = PROMPT_REGISTRY["query_rewrite"]
        _, user = tpl.render(input="它的原理是什么")
        assert "它的原理是什么" in user

    def test_rag_context_renders(self):
        tpl = PROMPT_REGISTRY["rag_context"]
        system, user = tpl.render(context="资料", input="问题")
        assert "资料" in user
        assert "问题" in user
