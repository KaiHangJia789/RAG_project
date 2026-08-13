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
    def test_registry_has_at_least_5_prompts(self):
        """验收要求：≥5 组 Prompt 对比"""
        assert len(PROMPT_REGISTRY) >= 5

    def test_required_prompt_names(self):
        required = ["baseline", "structured", "concise", "json_output", "chain_of_thought"]
        for name in required:
            assert name in PROMPT_REGISTRY, f"缺少 prompt: {name}"

    def test_all_prompts_renderable(self):
        for name, tpl in PROMPT_REGISTRY.items():
            system, user = tpl.render(input="测试输入")
            assert isinstance(system, str)
            assert "测试输入" in user
