"""
Tool 抽象与注册表

设计要点:
  - 工具定义**同时**产出两种形态：给模型看的 JSON Schema（`to_openai_schema()`）
    和给本地执行的实现（`run()`）。两者放在一起，避免"schema 改了实现没改"的漂移。
  - `run()` 收到的是**已解析的 dict**，不是 JSON 字符串 —— JSON 解析与容错
    统一在注册表层处理，工具实现只管业务逻辑。
  - **工具失败不抛异常**，返回错误描述字符串。理由：模型看到错误信息后
    可以自行纠正参数重试或换工具，而抛异常会中断整个 Agent 循环。
"""
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("rag_api.agent.tools")


@dataclass
class ToolResult:
    """工具执行结果"""

    ok: bool
    content: str                                  # 回传给模型的内容
    error: str | None = None
    meta: dict = field(default_factory=dict)      # 本地诊断用（延迟、命中数等）

    def to_message_content(self) -> str:
        """转成 tool 消息的 content（模型看到的就是这个字符串）"""
        return self.content


class Tool(ABC):
    """工具基类"""

    name: str = "tool"
    description: str = ""

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema 的 parameters 部分"""
        ...

    async def run(self, **kwargs: Any) -> ToolResult:
        """执行工具。子类实现 `_run`，本方法负责异常兜底。"""
        try:
            return await self._run(**kwargs)
        except TypeError as e:
            # 参数不匹配（模型给了 schema 之外的字段）——最常见，单独提示
            logger.warning("工具 %s 参数错误: %s", self.name, e)
            return ToolResult(
                ok=False,
                content=f"参数错误：{e}。请检查参数名与类型是否与工具定义一致。",
                error=str(e),
            )
        except Exception as e:
            logger.error("工具 %s 执行失败: %s", self.name, e)
            return ToolResult(
                ok=False,
                content=f"工具执行失败：{type(e).__name__}: {e}",
                error=str(e),
            )

    @abstractmethod
    async def _run(self, **kwargs: Any) -> ToolResult:
        ...

    def to_openai_schema(self) -> dict:
        """转成 OpenAI function calling 的工具定义"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """
    工具注册表 —— 导出 schema、按名分发执行。

    用法:
        reg = ToolRegistry([CalculatorTool(), SearchDocsTool(index)])
        tools_schema = reg.to_openai_schema()        # 传给 LLMClient.generate_chat(tools=...)
        result = await reg.execute("calculator", '{"expression": "1+1"}')
    """

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            logger.warning("工具 %s 被重复注册，后者覆盖前者", tool.name)
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def to_openai_schema(self) -> list[dict]:
        """全部工具的 OpenAI schema（传给模型）"""
        return [t.to_openai_schema() for t in self._tools.values()]

    async def execute(self, name: str, arguments: str | dict | None) -> ToolResult:
        """
        按名执行工具。

        Args:
            arguments: 模型给的参数 —— **通常是一段 JSON 字符串**（不是 dict！
            这是 openai function calling 的形态，见
            `ChatCompletionMessageFunctionToolCall.function.arguments`）。
            也可能是空串（无参数工具）或非法 JSON，都要容错。
        """
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self.names()) or "（无）"
            return ToolResult(
                ok=False,
                content=f"未知工具 '{name}'。可用工具：{available}",
                error="unknown_tool",
            )

        args = self._parse_arguments(arguments)
        if args is None:
            return ToolResult(
                ok=False,
                content=(
                    f"工具 '{name}' 的参数不是合法 JSON：{arguments!r}。"
                    f"请重新给出符合 schema 的 JSON 参数。"
                ),
                error="invalid_json",
            )

        return await tool.run(**args)

    @staticmethod
    def _parse_arguments(arguments: str | dict | None) -> dict | None:
        """
        解析工具参数。

        Returns:
            参数字典；解析失败返回 None（由调用方转成给模型的错误提示）
        """
        if arguments is None or arguments == "":
            return {}          # 无参数工具
        if isinstance(arguments, dict):
            return arguments   # 已经是 dict（部分 SDK 版本会直接给 dict）

        try:
            parsed = json.loads(arguments)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("工具参数 JSON 解析失败: %s | 原文: %r", e, arguments)
            return None

        if not isinstance(parsed, dict):
            logger.warning("工具参数不是对象: %r", parsed)
            return None
        return parsed
