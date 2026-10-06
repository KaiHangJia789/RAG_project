"""
ReAct 工具调用节点（第 13 周）

**ReAct 循环的形态**：
    agent（模型决定调不调工具）→ tools（执行工具）→ agent → ... → 生成答案

  与第 12 周"固定重写"的区别：重写角度与是否检索由**模型自主决定**，
  而不是预设的条件边。这才叫 Agent。

**关键实现约束**：
  1. 每轮把 assistant 消息（含 `tool_calls` 与 `reasoning_content`）**原样**加进
     消息流。漏掉 `reasoning_content` 会 HTTP 400（DeepSeek 在带 tools 时
     要求回传，见 scripts/spike_tool_calling.py 实测）。
     所以统一用 `ChatMessage.from_response()` 构造，禁止手写 dict。
  2. 工具执行失败**不中断循环** —— 把错误信息作为工具结果回传，让模型
     自行纠正或换工具。
  3. 轮次上限 `AGENT_TOOL_MAX_ROUNDS` 是硬约束，防止模型反复调同一个工具。
"""
import logging
from typing import Any

from app.agent.nodes import config_of
from app.agent.serialization import hits_to_dicts
from app.agent.state import AgentState
from app.config import settings
from app.llm.client import ChatMessage, LLMClient
from app.agent.tools import ToolRegistry

logger = logging.getLogger("rag_api.agent.tools_node")

# 工具调用阶段使用的 system prompt
_TOOL_AGENT_SYSTEM = """你是一个可以调用工具的问答助手。

可用工具：
{tools_desc}

【输出格式 — 必须严格遵守】
最终答案必须是**纯文本段落**，并遵守以下规则：
1. 每个事实性陈述后紧跟引用编号，格式 [1]、[2]，可连用如 [1][3]。
   编号对应 search_documents 返回结果里的 [n] 标号；不要编造编号。
2. **禁止使用 Markdown**：不要用 # 标题、不要用 | 表格、不要用 ** 加粗、
   不要用 $$ 公式块、不要用代码块。只用普通句子和换行。
3. 不要写"根据工具返回""以下是"之类的开场白，直接给答案。

【工作方式】
1. 需要知识库内容 → 调用 search_documents。
2. 需要算术计算 → calculator；需要知识库统计 → query_metadata。
3. 一次可调用多个互不依赖的工具。
4. 收集到足够信息后直接给出最终答案，不要再调用工具。
5. 所有工具都没找到答案时，只回复：根据已有资料无法回答该问题。
"""


def make_tool_agent_node(bundle, registry: ToolRegistry):
    """
    ReAct 的"思考+决策"节点：让模型决定调工具还是给最终答案。

    该节点不直接产出最终答案 —— 它把模型的响应（含 tool_calls）写进 messages，
    由条件边决定是去执行工具还是收尾。
    """

    async def agent(state: AgentState) -> dict:
        messages = list(state.get("messages") or [])

        # 首次进入：构造 system + user
        if not messages:
            tools_desc = "\n".join(
                f"- {t.name}: {t.description}" for t in
                (registry.get(n) for n in registry.names()) if t
            )
            messages = [
                {"role": "system", "content": _TOOL_AGENT_SYSTEM.format(tools_desc=tools_desc)},
                {"role": "user", "content": state.get("original_question", "")},
            ]

        resp = await bundle.llm.generate_chat(
            messages,
            tools=registry.to_openai_schema(),
            tool_choice="auto",
            thinking_enabled=False,     # 工具轮关思考：快、稳、便宜（spike 已验证可共存）
        )

        # ── 关键：用工厂构造 assistant 消息，保证 reasoning_content 被保留 ──
        assistant = ChatMessage.from_response(resp).to_openai_dict()

        update: dict = {
            "messages": [assistant],
            "tool_calls": resp.tool_calls or [],
            "llm_call_count": (state.get("llm_call_count") or 0) + 1,
            "prompt_tokens": (state.get("prompt_tokens") or 0) + resp.usage.prompt_tokens,
            "completion_tokens": (
                (state.get("completion_tokens") or 0) + resp.usage.completion_tokens
            ),
        }

        if resp.tool_calls:
            logger.info(
                "Agent 请求 %d 个工具: %s",
                len(resp.tool_calls),
                [tc.get("function", {}).get("name") for tc in resp.tool_calls],
            )
        else:
            # 没有工具调用 → 这就是最终答案
            logger.info("Agent 给出最终答案（无工具调用）")
            update["answer"] = resp.text or ""

        return update

    return agent


def make_tool_executor_node(bundle, registry: ToolRegistry):
    """
    ReAct 的"行动"节点：执行模型请求的（多个）工具。

    结果以 `role="tool"` 消息回传，`tool_call_id` 必须与请求一一对应 ——
    对不上模型会困惑或报错。
    """

    async def execute(state: AgentState) -> dict:
        tool_calls = state.get("tool_calls") or []
        if not tool_calls:
            return {}

        messages: list[dict] = []
        results: list[dict] = []
        # 累积检索命中：每次 search_documents 调用返回的原始结果
        collected_hits: list = []

        for tc in tool_calls:
            fn = tc.get("function", {}) or {}
            name = fn.get("name", "")
            args_raw = fn.get("arguments", "")
            call_id = tc.get("id", "")

            # registry.execute 内部处理 JSON 解析失败与执行异常，
            # 始终返回 ToolResult（不抛异常），所以循环不会被单次失败打断
            result = await registry.execute(name, args_raw)

            # 检索工具把原始命中放在 meta.raw_hits —— 直接取用，
            # 不做关键词二次回捞（那样既不准又脆弱）
            raw = (result.meta or {}).get("raw_hits")
            if raw:
                collected_hits.extend(raw)

            messages.append(
                ChatMessage.tool_result(call_id, result.content).to_openai_dict()
            )
            results.append({
                "tool_call_id": call_id,
                "name": name,
                "arguments": args_raw,
                "ok": result.ok,
                "content": result.content[:2000],     # 截断，避免状态膨胀
                "error": result.error,
            })

            logger.info(
                "工具 %s %s: %s",
                name, "成功" if result.ok else "失败", result.content[:80],
            )

        update: dict = {
            "messages": messages,
            "tool_results": results,
            "tool_round": (state.get("tool_round") or 0) + 1,
            "tool_calls": [],      # 清空，避免条件边误判为"还有工具要执行"
        }

        # 有检索命中则写进 state，供 citations 节点做引用映射
        if collected_hits:
            update["hits"] = _dedupe_hits(collected_hits)

        return update

    return execute


def _dedupe_hits(raw_hits: list) -> list[dict]:
    """
    对多次检索的命中按 faiss_id 去重（保留首次出现的顺序），转成 state 格式。

    为什么要去重：Agent 可能用不同措辞检索多次，同一条 chunk 会被命中多次。
    不去重会让引用列表出现重复项，且 [n] 编号与 citations 数组的对应关系错乱。
    """
    seen: set[int] = set()
    unique: list = []
    for h in raw_hits:
        if h.faiss_id in seen:
            continue
        seen.add(h.faiss_id)
        unique.append(h)

    # 按相似度降序（多次检索的结果混在一起，需要重新排序才能正确编号）
    unique.sort(key=lambda x: -x.score)
    return hits_to_dicts(unique)


def make_finalize_node(bundle):
    """
    收尾节点：用**标准引用管线**重新生成最终答案。

    **为什么要多这一步**：
      实测发现 ReAct 模式下模型倾向自由发挥 —— 即便在 system prompt 里
      反复要求「用 [n] 标注引用、禁用 Markdown」，它仍会输出带 ## 标题和
      ** 加粗的自由文本，引用率为 0。而固定管线的 `rag_context_cited`
      prompt 能稳定产出结构化引用。

      解法是**混合架构**：让 Agent 循环负责"灵活地收集信息"（这是它的强项），
      把"按格式产出带引用的答案"交回经过验证的标准管线（这是它的强项）。

      代价是额外一次 LLM 调用 —— 但换来可用的引用，值得。
      当检索命中为空（例如只用 calculator/query_metadata 的场景）时跳过本步，
      直接沿用 Agent 的自由文本答案。
    """

    async def finalize(state: AgentState) -> dict:
        from app.agent.serialization import dicts_to_hits
        from app.rag import steps

        hits = dicts_to_hits(state.get("hits") or [])
        answer = state.get("answer") or ""
        question = state.get("original_question") or state.get("question", "")

        # 没有检索命中 → 无法走引用管线（例如纯计算/元数据问题），保持原答案
        if not hits:
            return {}

        # 已经有合规引用 → 不必重生成（省一次调用）
        from app.rag.citations import extract_citation_indices

        if extract_citation_indices(answer):
            return {}

        cfg = config_of(state)
        context = steps.assemble_context(hits)

        try:
            resp = await steps.generate_answer(bundle.llm, question, context, config=cfg)
        except Exception as e:
            logger.warning("收尾重生成失败，沿用 Agent 原答案: %s", e)
            return {}

        logger.info("收尾：用标准引用管线重生成答案")

        return {
            "answer": resp.text or answer,
            "llm_call_count": (state.get("llm_call_count") or 0) + 1,
            "prompt_tokens": (state.get("prompt_tokens") or 0) + resp.usage.prompt_tokens,
            "completion_tokens": (
                (state.get("completion_tokens") or 0) + resp.usage.completion_tokens
            ),
        }

    return finalize


def _route_after_agent(state: AgentState) -> str:
    """
    条件边：有工具调用 → 执行工具；否则 → 收尾。

    同时检查轮次上限 —— 超限时强制收尾，防止模型反复调同一个工具。
    """
    rounds = state.get("tool_round") or 0
    if rounds >= settings.AGENT_TOOL_MAX_ROUNDS:
        logger.warning("工具调用达上限 %d，强制收尾", rounds)
        return "finish"

    return "tools" if state.get("tool_calls") else "finish"
