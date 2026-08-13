"""
LLM 客户端测试（不调用真实 API，mock 掉 openai）
"""
import pytest
from pydantic import BaseModel

from app.llm.client import (
    ChatMessage,
    Conversation,
    LLMClient,
    LLMResponse,
    LLMUsage,
)


class TestLLMUsage:
    def test_defaults(self):
        usage = LLMUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_fields(self):
        usage = LLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.total_tokens == 30


class TestChatMessage:
    def test_basic_to_openai_dict(self):
        msg = ChatMessage(role="user", content="hello")
        assert msg.to_openai_dict() == {"role": "user", "content": "hello"}

    def test_reasoning_content_preserved(self):
        """关键测试：reasoning_content 必须原样回传（DeepSeek 多轮陷阱）"""
        msg = ChatMessage(
            role="assistant",
            content="答案",
            reasoning_content="思考过程...",
        )
        d = msg.to_openai_dict()
        assert d["reasoning_content"] == "思考过程..."
        assert d["content"] == "答案"

    def test_tool_calls_preserved(self):
        msg = ChatMessage(
            role="assistant",
            content="",
            tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}}],
        )
        d = msg.to_openai_dict()
        assert d["tool_calls"][0]["id"] == "call_1"


class TestConversation:
    def test_add_messages(self):
        conv = Conversation()
        conv.add_user("问题")
        conv.add_assistant("回答", reasoning_content="思考")
        assert len(conv) == 2

    def test_reasoning_content_survives_roundtrip(self):
        """多轮对话中 reasoning_content 不丢失"""
        conv = Conversation()
        conv.add_user("第一问")
        conv.add_assistant("第一答", reasoning_content="推理1")
        conv.add_user("第二问")

        msgs = conv.to_openai_messages()
        # 第二条（assistant）的 reasoning_content 必须保留
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["reasoning_content"] == "推理1"
        # 第三条（user）没有 reasoning_content
        assert "reasoning_content" not in msgs[2]


class TestLLMClientBuildParams:
    def _make_client(self):
        client = LLMClient.__new__(LLMClient)  # 跳过 __init__（不建真实连接）
        client._model = "deepseek-v4-pro"
        client._default_max_tokens = 16000
        client._default_reasoning_effort = "high"
        client._default_thinking = True
        return client

    def test_build_params_defaults(self):
        client = self._make_client()
        params = client._build_params(
            [{"role": "user", "content": "hi"}],
            max_tokens=None,
            reasoning_effort=None,
            thinking_enabled=None,
        )
        assert params["model"] == "deepseek-v4-pro"
        assert params["max_tokens"] == 16000
        assert params["reasoning_effort"] == "high"
        # DeepSeek 思考模式通过 extra_body 控制
        assert params["extra_body"] == {"thinking": {"type": "enabled"}}

    def test_build_params_override(self):
        client = self._make_client()
        params = client._build_params(
            [{"role": "user", "content": "hi"}],
            max_tokens=32000,
            reasoning_effort="max",
            thinking_enabled=False,
        )
        assert params["max_tokens"] == 32000
        assert params["reasoning_effort"] == "max"
        assert params["extra_body"] == {"thinking": {"type": "disabled"}}

    def test_build_params_max_tokens_not_too_low(self):
        """用户要求：max_tokens 至少 16000，不能太低"""
        client = self._make_client()
        params = client._build_params([{"role": "user", "content": "x"}],
                                      max_tokens=None, reasoning_effort=None,
                                      thinking_enabled=None)
        assert params["max_tokens"] >= 16000
