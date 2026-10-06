"""
LLM 客户端 — DeepSeek V4 Pro（OpenAI 兼容协议）

关键设计:
  - 使用 openai SDK（DeepSeek 通过 base_url 接入）
  - 思考模式通过 reasoning_effort + extra_body 控制（DeepSeek 专属，非 Claude thinking 参数）
  - reasoning_content 字段预留 + 多轮历史保留（Week 13-15 Agentic RAG 必需，避免后期重构）
  - LangFuse 自动追踪：有 key 用 langfuse.openai 包装器，无 key 降级纯 openai
"""
import time
import logging
from typing import Any

from pydantic import BaseModel, Field

from app.config import settings
from app.llm.observability import langfuse_enabled

logger = logging.getLogger("rag_api.llm")


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

class LLMUsage(BaseModel):
    """token 用量（OpenAI 协议字段名）"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """统一 LLM 响应"""
    text: str
    model: str
    finish_reason: str = "stop"
    usage: LLMUsage = Field(default_factory=LLMUsage)
    latency_ms: float = 0.0
    reasoning_content: str | None = None   # ← DeepSeek 思考内容，多轮时必须原样回传
    # 模型请求的工具调用（Week13）。有值时 finish_reason 通常是 "tool_calls"。
    # 结构：[{"id": "call_xxx", "type": "function", "function": {"name": ..., "arguments": "<json 字符串>"}}]
    tool_calls: list[dict] | None = None


class ChatMessage(BaseModel):
    """
    单条对话消息。

    reasoning_content: DeepSeek V4 Pro 思考模式下，assistant 消息会携带此字段。
    工具调用场景下，后续请求必须原封不动回传，否则 HTTP 400。
    这里显式建模，避免 Week 13-15 做 Agentic RAG 时重构。
    """
    role: str
    content: str
    reasoning_content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_openai_dict(self) -> dict:
        """转为 OpenAI API 可接受的 dict，保留 reasoning_content"""
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.reasoning_content is not None:
            d["reasoning_content"] = self.reasoning_content   # ← 关键：原样回传
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d

    # ── 工厂方法 ────────────────────────────────────────────

    @classmethod
    def from_response(cls, resp: "LLMResponse") -> "ChatMessage":
        """
        把 LLM 响应转成 assistant 消息（供下一轮回传）。

        **必须用这个工厂构造，不要手写 dict。** DeepSeek 在带 tools 的请求里
        要求 assistant 消息的 `reasoning_content` 原样回传（即使该轮没调工具），
        手写 dict 时极易漏掉这个字段，导致 HTTP 400 且报错信息晦涩。

        ⚠️ 注意：这与 langchain 的 `ChatDeepSeek` 是同一类 bug —— 它会丢掉
        该字段（其 PR #40254 才修复）。这也是本项目不用 langchain-openai
        而自己扩展客户端的原因之一。
        """
        return cls(
            role="assistant",
            content=resp.text or "",
            reasoning_content=resp.reasoning_content,
            tool_calls=resp.tool_calls,
        )

    @classmethod
    def tool_result(cls, tool_call_id: str, content: str) -> "ChatMessage":
        """构造工具执行结果消息（回传给模型）"""
        return cls(role="tool", content=content, tool_call_id=tool_call_id)


class Conversation:
    """多轮对话历史，正确保留 reasoning_content"""

    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []

    def add_user(self, content: str) -> None:
        self.messages.append(ChatMessage(role="user", content=content))

    def add_assistant(self, content: str, reasoning_content: str | None = None) -> None:
        self.messages.append(ChatMessage(
            role="assistant",
            content=content,
            reasoning_content=reasoning_content,
        ))

    def to_openai_messages(self) -> list[dict]:
        return [m.to_openai_dict() for m in self.messages]

    def __len__(self) -> int:
        return len(self.messages)


# ═══════════════════════════════════════════════════════════════
# LLM 客户端
# ═══════════════════════════════════════════════════════════════

class LLMClient:
    """
    DeepSeek V4 Pro 客户端。

    Usage:
        client = LLMClient()
        resp = await client.generate(
            system="你是一个助手",
            user_message="什么是 RAG？",
        )
        print(resp.text)
        print(resp.usage.total_tokens)
    """

    def __init__(self) -> None:
        self._model = settings.DEEPSEEK_MODEL
        self._default_max_tokens = settings.LLM_MAX_TOKENS
        self._default_reasoning_effort = settings.LLM_REASONING_EFFORT
        self._default_thinking = settings.LLM_THINKING_ENABLED

        # LangFuse 追踪：有 key 用包装器，无 key 降级纯 openai
        if langfuse_enabled():
            from langfuse.openai import AsyncOpenAI
            self._client: Any = AsyncOpenAI(
                api_key=settings.DEEPSEEK_API_KEY,
                base_url=settings.DEEPSEEK_BASE_URL,
            )
            logger.info("LLMClient: 使用 LangFuse 追踪（langfuse.openai 包装器）")
        else:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=settings.DEEPSEEK_API_KEY,
                base_url=settings.DEEPSEEK_BASE_URL,
            )
            logger.info("LLMClient: LangFuse 未配置，使用纯 openai 客户端")

    # ═══════════════════════════════════════════════════════════════
    # 单次生成
    # ═══════════════════════════════════════════════════════════════

    async def generate(
        self,
        system: str,
        user_message: str,
        *,
        history: list[dict] | None = None,      # 之前的多轮消息（含 reasoning_content）
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        thinking_enabled: bool | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """
        单次生成。

        Args:
            system: 系统提示
            user_message: 用户消息
            history: 多轮历史（OpenAI 格式 dict 列表，reasoning_content 会原样保留）
            max_tokens: 覆盖默认值
            reasoning_effort: low / high / max
            thinking_enabled: 是否开启思考模式
            temperature: 采样温度。**仅在 thinking_enabled=False 时传** ——
                推理模型开启思考时采样参数会被忽略甚至报错，传了没意义还有风险。
                判官调用应传 0.0 保证可复现。
            json_mode: 要求返回 JSON 对象（response_format=json_object）。
                **仅在 thinking_enabled=False 时用**；且 prompt 里必须出现字面量 "json"，
                这是 DeepSeek json_object 模式的要求。
        """
        # 1. 组装消息
        messages: list[dict] = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        # 2. 组装参数
        params = self._build_params(
            messages,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            thinking_enabled=thinking_enabled,
            temperature=temperature,
            json_mode=json_mode,
        )

        # 3. 委托给 generate_chat（统一调用路径）
        return await self.generate_chat(
            messages,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            thinking_enabled=thinking_enabled,
            temperature=temperature,
            json_mode=json_mode,
        )

    async def generate_chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        thinking_enabled: bool | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> LLMResponse:
        """
        按完整消息列表生成（Agent 循环用）。

        **与 `generate()` 的分工**：
          `generate(system, user_message, history=...)` 适合"一问一答 + 历史"的形态；
          但 Agent 循环里的消息流是混杂的 ——
          system + user + assistant(带 tool_calls) + tool(结果) + assistant(再调工具)...
          用 system/user 两段式拼不出来，所以需要这个直接传 messages 的入口。

        Args:
            messages: 完整消息列表（OpenAI 格式 dict）。
                用 `ChatMessage.to_openai_dict()` 构造，**不要手写 dict** ——
                手写极易漏掉 assistant 消息的 `reasoning_content`，
                而 DeepSeek 在带 tools 时要求该字段原样回传，漏了就 HTTP 400。
            tools: 工具定义列表（OpenAI function calling 格式）
            tool_choice: "auto" / "none" / "required" / {"type":"function",...}
            parallel_tool_calls: 是否允许并行工具调用

        Returns:
            LLMResponse；若模型请求调用工具，`tool_calls` 非空且
            `finish_reason` 通常为 "tool_calls"
        """
        params = self._build_params(
            messages,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            thinking_enabled=thinking_enabled,
            temperature=temperature,
            json_mode=json_mode,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )

        start = time.monotonic()
        resp = await self._client.chat.completions.create(**params)
        elapsed_ms = (time.monotonic() - start) * 1000

        return self._parse_response(resp, elapsed_ms)

    # ═══════════════════════════════════════════════════════════════
    # 参数组装（含 DeepSeek 思考模式控制）
    # ═══════════════════════════════════════════════════════════════

    def _build_params(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None,
        reasoning_effort: str | None,
        thinking_enabled: bool | None,
        temperature: float | None = None,
        json_mode: bool = False,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> dict:
        """组装 OpenAI 调用参数，正确处理 DeepSeek 思考模式"""
        params: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens or self._default_max_tokens,
        }

        # reasoning_effort: low / high / max（DeepSeek 顶层参数）
        effort = reasoning_effort or self._default_reasoning_effort
        if effort:
            params["reasoning_effort"] = effort

        # thinking: enabled / disabled（DeepSeek 扩展参数，走 extra_body）
        enabled = self._default_thinking if thinking_enabled is None else thinking_enabled
        params["extra_body"] = {
            "thinking": {"type": "enabled" if enabled else "disabled"}
        }

        # 仅在显式传入时才加 —— 不用默认值填充，避免改变既有行为
        # （现有测试断言 extra_body 精确相等，且未传这两个参数时不该出现）
        if temperature is not None:
            params["temperature"] = temperature
        if json_mode:
            params["response_format"] = {"type": "json_object"}

        # ── 工具调用（Week13）──
        # 全部"仅显式传入才加"：不自动填 tool_choice="auto"，
        # 否则等于改变了所有既有调用的请求形态。
        # （tools 与 thinking 可共存 —— 已由 scripts/spike_tool_calling.py 实测验证）
        if tools:
            params["tools"] = tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            params["parallel_tool_calls"] = parallel_tool_calls

        return params

    # ═══════════════════════════════════════════════════════════════
    # 响应解析（含 reasoning_content 提取）
    # ═══════════════════════════════════════════════════════════════

    def _parse_response(self, resp: Any, elapsed_ms: float) -> LLMResponse:
        """OpenAI ChatCompletion → LLMResponse"""
        choice = resp.choices[0]
        message = choice.message

        text = message.content or ""
        # DeepSeek 思考内容：openai SDK 不类型化此字段，用 getattr 安全提取
        reasoning = getattr(message, "reasoning_content", None)
        finish_reason = choice.finish_reason or "stop"

        # ── 工具调用提取（Week13）──
        # openai 3.0 的对象形态（已实测）：
        #   ChatCompletionMessageFunctionToolCall
        #     ├─ id: str
        #     ├─ type: Literal["function"]
        #     └─ function: Function{ name: str, arguments: str }   ← arguments 是 JSON 字符串
        # 转成纯 dict 后再往上传，避免上层依赖 SDK 的内部类型
        # （SDK 大版本升级时对象结构会变，dict 是稳定契约）。
        raw_tcs = getattr(message, "tool_calls", None)
        tool_calls: list[dict] | None = None
        if raw_tcs:
            tool_calls = []
            for tc in raw_tcs:
                try:
                    tool_calls.append(tc.model_dump())
                except AttributeError:
                    # 退化路径：SDK 对象没有 model_dump（极旧版本或 FutureWarning 包装）
                    tool_calls.append({
                        "id": getattr(tc, "id", ""),
                        "type": getattr(tc, "type", "function"),
                        "function": {
                            "name": getattr(getattr(tc, "function", None), "name", ""),
                            "arguments": getattr(
                                getattr(tc, "function", None), "arguments", "{}"
                            ),
                        },
                    })

        usage = resp.usage
        return LLMResponse(
            text=text,
            reasoning_content=reasoning,
            model=resp.model or self._model,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            usage=LLMUsage(
                prompt_tokens=usage.prompt_tokens or 0,
                completion_tokens=usage.completion_tokens or 0,
                total_tokens=usage.total_tokens or 0,
            ),
            latency_ms=round(elapsed_ms, 2),
        )
