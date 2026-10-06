"""
T1 Spike — 实测 DeepSeek 思考模式与工具调用的兼容性

这是第 11-13 周开工前的**准入门槛**。结论决定第 13 周 Agent 循环的设计。

要回答四个问题:
  1. `thinking=enabled` + `tools` 能否共存？（若不能，Agent 必须关思考）
  2. 多轮工具调用时 `reasoning_content` 的回传规则？（DeepSeek 文档说带 tools
     时**必须**回传，否则 400；需实测确认）
  3. openai 3.0 SDK 里 `tool_calls` 的实际形态（已知 arguments 是 JSON 字符串）
  4. 工具结果的正确回传格式（role=tool + tool_call_id）

用法:
    python scripts/spike_tool_calling.py
"""
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import settings  # noqa: E402
from app.llm.observability import langfuse_enabled  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s | %(message)s")


# 一个简单工具：计算器
CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "计算数学表达式。当用户需要做算术运算时调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式，例如 '123 * 456'",
                }
            },
            "required": ["expression"],
        },
    },
}


def execute_tool(name: str, arguments: str) -> str:
    """
    执行工具。注意 arguments 是 **JSON 字符串**（不是 dict），必须解析。

    模型可能给出非法 JSON 或空串，所以解析要容错。
    """
    try:
        args = json.loads(arguments) if arguments.strip() else {}
    except json.JSONDecodeError as e:
        return f"参数解析失败: {e}"

    if name == "calculator":
        expr = args.get("expression", "")
        try:
            # 仅用于 spike 演示；生产实现不用 eval（见 app/agent/tools/calculator.py）
            result = eval(expr, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"计算失败: {e}"

    return f"未知工具: {name}"


def make_client() -> AsyncOpenAI:
    if langfuse_enabled():
        from langfuse.openai import AsyncOpenAI as LFOpenAI
        return LFOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
        )
    return AsyncOpenAI(
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
    )


async def call(client, messages, *, tools=None, thinking: bool):
    """单次调用，返回完整的 response 对象"""
    params = {
        "model": settings.DEEPSEEK_MODEL,
        "messages": messages,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "extra_body": {"thinking": {"type": "enabled" if thinking else "disabled"}},
    }
    if tools:
        params["tools"] = tools
    return await client.chat.completions.create(**params)


def describe_response(resp) -> dict:
    """把一个响应拆解成可打印的结构"""
    msg = resp.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None)
    return {
        "content": (msg.content or "")[:100],
        "finish_reason": resp.choices[0].finish_reason,
        "reasoning_content": (getattr(msg, "reasoning_content", None) or "")[:80] or None,
        "tool_calls_count": len(tool_calls) if tool_calls else 0,
        "tool_calls_raw_type": type(tool_calls[0]).__name__ if tool_calls else None,
        "tool_calls": [tc.model_dump() for tc in tool_calls] if tool_calls else None,
    }


# ═══════════════════════════════════════════════════════════════
# 场景 1：思考模式 + tools 能否共存
# ═══════════════════════════════════════════════════════════════

async def test_thinking_with_tools(client) -> dict:
    print("=" * 72)
    print("场景 1：thinking=enabled + tools 共存性")
    print("=" * 72)

    messages = [{"role": "user", "content": "请用计算器算一下 123 乘以 456 等于多少。"}]

    try:
        resp = await call(client, messages, tools=[CALCULATOR_TOOL], thinking=True)
        info = describe_response(resp)
        print(f"  ✅ 调用成功（thinking=enabled + tools）")
        print(f"     finish_reason: {info['finish_reason']}")
        print(f"     tool_calls 数量: {info['tool_calls_count']}")
        print(f"     reasoning_content: {'有' if info['reasoning_content'] else '无'}")
        if info["tool_calls"]:
            print(f"     tool_calls 结构类型: {info['tool_calls_raw_type']}")
            print(f"     raw: {json.dumps(info['tool_calls'], ensure_ascii=False)[:200]}")
        return {"ok": True, "info": info, "messages": messages}
    except Exception as e:
        print(f"  ❌ 调用失败: {type(e).__name__}: {str(e)[:300]}")
        return {"ok": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════
# 场景 2：多轮工具调用 + reasoning_content 回传规则
# ═══════════════════════════════════════════════════════════════

async def test_multi_turn(client, thinking: bool) -> None:
    label = "开启" if thinking else "关闭"
    print()
    print("=" * 72)
    print(f"场景 2：多轮工具调用（thinking={label}）+ reasoning_content 回传")
    print("=" * 72)

    messages = [{"role": "user", "content": "请用计算器算一下 123 乘以 456 等于多少。"}]

    try:
        # ── 第 1 轮：期望模型请求调用工具 ──
        resp1 = await call(client, messages, tools=[CALCULATOR_TOOL], thinking=thinking)
        msg1 = resp1.choices[0].message
        tcs = getattr(msg1, "tool_calls", None)

        if not tcs:
            print(f"  ⚠️  模型没调用工具，直接回答: {(msg1.content or '')[:80]}")
            print("     （这不影响兼容性结论，但说明该问题太简单）")
            return

        print(f"  第 1 轮: 模型请求调用 {len(tcs)} 个工具")
        tc = tcs[0]
        print(f"     工具名: {tc.function.name}")
        print(f"     arguments 原始值: {tc.function.arguments!r}  (类型 {type(tc.function.arguments).__name__})")

        result = execute_tool(tc.function.name, tc.function.arguments)
        print(f"     本地执行结果: {result}")

        # ── 组装第 2 轮消息（关键：assistant 消息必须带 reasoning_content）──
        assistant_msg = {"role": "assistant", "content": msg1.content or ""}
        reasoning = getattr(msg1, "reasoning_content", None)
        if reasoning is not None:
            assistant_msg["reasoning_content"] = reasoning
        assistant_msg["tool_calls"] = [tc.model_dump() for tc in tcs]
        messages.append(assistant_msg)
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result,
        })

        print(f"  第 2 轮请求携带的 assistant 消息字段: {list(assistant_msg.keys())}")

        # ── 第 2 轮：带 reasoning_content 回传 ──
        resp2 = await call(client, messages, tools=[CALCULATOR_TOOL], thinking=thinking)
        final = resp2.choices[0].message.content or ""
        print(f"  ✅ 第 2 轮成功（reasoning_content 已回传）")
        print(f"     最终回答: {final[:120]}")

        # ── 对照实验：故意不回传 reasoning_content ──
        if reasoning is not None:
            print()
            print("  对照实验：故意**不**回传 reasoning_content")
            msg_no_reasoning = {"role": "assistant", "content": msg1.content or ""}
            msg_no_reasoning["tool_calls"] = [tc.model_dump() for tc in tcs]
            msgs2 = messages[:1] + [msg_no_reasoning, {
                "role": "tool", "tool_call_id": tc.id, "content": result,
            }]
            try:
                await call(client, msgs2, tools=[CALCULATOR_TOOL], thinking=thinking)
                print("     ⚠️  不传 reasoning_content 也成功了 ——")
                print("        说明当前模型/版本对该字段是宽松的")
            except Exception as e:
                print(f"     ✅ 如文档所述报错（证明必须回传）: {type(e).__name__}")
                print(f"        {str(e)[:200]}")

    except Exception as e:
        print(f"  ❌ 失败: {type(e).__name__}: {str(e)[:300]}")


async def main() -> None:
    print()
    print("DeepSeek 工具调用兼容性 Spike")
    print(f"模型: {settings.DEEPSEEK_MODEL}")
    print(f"LangFuse 追踪: {'开启' if langfuse_enabled() else '关闭'}")
    print()

    client = make_client()

    r1 = await test_thinking_with_tools(client)
    await test_multi_turn(client, thinking=True)
    await test_multi_turn(client, thinking=False)

    print()
    print("=" * 72)
    print("结论")
    print("=" * 72)
    if r1["ok"]:
        print("  ✅ thinking=enabled 与 tools 可以共存")
        print("     → 第 13 周 Agent 循环可以保留思考模式")
    else:
        print("  ❌ thinking=enabled 与 tools 冲突")
        print("     → 第 13 周 Agent 必须 thinking_enabled=False，设计需调整")


if __name__ == "__main__":
    asyncio.run(main())
