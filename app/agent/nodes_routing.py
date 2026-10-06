"""
第 12 周节点 — 查询分类、相关性评分、查询重写

这三个节点让工作流从「直线」变成「有分支、有循环」：
  - classify 提供**分支依据**（条件边的判定输入）
  - grade 提供**循环依据**（不达标才重写）
  - rewrite 是**循环的修复手段**

设计要点:
  - 全部**关闭思考模式**：这三类调用都是结构化决策（要 JSON 输出），
    关思考既快又便宜，且 json_mode 要求 thinking_enabled=False。
  - 全部**容错降级**：判官式调用可能返回非法 JSON 或超时，
    失败时走向保守路径（分类默认 factual、评分默认达标、重写默认不变），
    绝不让整条链路因为一次辅助调用失败而崩掉。
  - 复用 `parse_llm_json`（judge.py 里已有的容错解析），不重复造轮子。
"""
import logging

from app.agent.prompts import get_agent_prompt
from app.agent.serialization import dicts_to_hits
from app.agent.state import AgentState
from app.evaluation.judge import parse_llm_json

logger = logging.getLogger("rag_api.agent.routing")

VALID_QUERY_TYPES = ("factual", "reasoning", "chitchat")


def usage_update(state: AgentState, resp=None, *, extra: dict | None = None) -> dict:
    """
    构造「调用次数 + token 用量」的状态更新。

    统一成一处，避免每个节点各写一遍而漏掉 token 累计 ——
    第 13 周的成本对比依赖这个数据的完整性，漏一个节点就会低报成本，
    而且很难发现（数字看起来是合理的，只是偏小）。

    Args:
        resp: LLMResponse；None 表示调用失败，只累加次数不加 token
        extra: 该节点特有的其它字段
    """
    update: dict = {
        "llm_call_count": (state.get("llm_call_count") or 0) + 1,
    }
    if resp is not None:
        update["prompt_tokens"] = (
            (state.get("prompt_tokens") or 0) + resp.usage.prompt_tokens
        )
        update["completion_tokens"] = (
            (state.get("completion_tokens") or 0) + resp.usage.completion_tokens
        )
    if extra:
        update.update(extra)
    return update


# ═══════════════════════════════════════════════════════════════
# 节点：查询分类
# ═══════════════════════════════════════════════════════════════

def make_classify_node(bundle):
    """
    把问题分成 factual / reasoning / chitchat，供条件边路由。

    降级策略：调用失败或返回非法值时**默认 factual** ——
    宁可多检索一次，也不要把真实问题误判成闲聊而不检索。
    """

    async def classify(state: AgentState) -> dict:
        question = state.get("original_question") or state.get("question", "")
        template = get_agent_prompt("agent_query_classify")
        system, user_msg = template.render(question=question)

        resp = None
        qtype = "factual"       # 降级默认值
        try:
            resp = await bundle.llm.generate(
                system=system,
                user_message=user_msg,
                thinking_enabled=False,
                temperature=0.0,
                json_mode=True,
            )
            data = parse_llm_json(resp.text or "")
            candidate = str(data.get("query_type", "")).strip().lower()

            if candidate in VALID_QUERY_TYPES:
                qtype = candidate
            else:
                logger.warning("分类返回非法值 %r，降级为 factual", candidate)

        except Exception as e:
            logger.warning("查询分类失败，降级为 factual: %s", e)

        logger.info("查询分类: %s → %s", question[:30], qtype)

        return usage_update(state, resp, extra={"query_type": qtype})

    return classify


# ═══════════════════════════════════════════════════════════════
# 节点：相关性评分
# ═══════════════════════════════════════════════════════════════

def make_grade_node(bundle):
    """
    评估检索到的上下文能否回答问题，写入 `relevance_score`。

    降级策略：调用失败时**默认 1.0（视为达标）** ——
    宁可带着当前上下文去生成（可能拒答），也不要因为评分服务抖动
    触发无意义的重写循环。
    """

    async def grade(state: AgentState) -> dict:
        hits = state.get("hits") or []
        question = state.get("original_question") or state.get("question", "")

        # 没有检索结果 → 0 分（触发重写或直接拒答）
        if not hits:
            return {"relevance_score": 0.0}

        context = state.get("context") or ""
        template = get_agent_prompt("agent_relevance_grade")
        system, user_msg = template.render(question=question, context=context)

        resp = None
        score, missing = 1.0, ""        # 降级：视为达标，避免无意义重写
        try:
            resp = await bundle.llm.generate(
                system=system,
                user_message=user_msg,
                thinking_enabled=False,
                temperature=0.0,
                json_mode=True,
            )
            data = parse_llm_json(resp.text or "")
            score = max(0.0, min(1.0, float(data.get("score", 1.0))))
            missing = str(data.get("missing", "") or "")

        except Exception as e:
            logger.warning("相关性评分失败，降级为达标(1.0): %s", e)

        logger.info(
            "相关性评分: %.2f%s",
            score, f" (缺: {missing[:40]})" if missing else "",
        )

        return usage_update(state, resp, extra={
            "relevance_score": score,
            # 供重写节点使用（LangGraph 的 state 是共享的）
            "relevance_missing": missing,
        })

    return grade


# ═══════════════════════════════════════════════════════════════
# 节点：查询重写
# ═══════════════════════════════════════════════════════════════

def make_rewrite_node(bundle):
    """
    改写查询以提升召回。

    **语义收敛保护**：若改写结果与上一次查询相同（模型没能给出新角度），
    直接把 rewrite_count 推到上限强制跳出 —— 否则会在
    「检索→评分不变→改写不变→再检索」上空转到递归上限。
    这是三重终止条件里的第三重（另两重是次数上限与 recursion_limit）。
    """

    async def rewrite(state: AgentState) -> dict:
        question = state.get("original_question") or state.get("question", "")
        last_query = state.get("question", "")
        missing = state.get("relevance_missing", "")
        max_rewrites = state.get("max_rewrites", 2)
        count = (state.get("rewrite_count") or 0) + 1

        template = get_agent_prompt("agent_query_rewrite")
        system, user_msg = template.render(
            question=question, last_query=last_query, missing=missing or "未说明"
        )

        resp = None
        new_query = last_query      # 降级：改写失败则保持原查询
        try:
            resp = await bundle.llm.generate(
                system=system,
                user_message=user_msg,
                thinking_enabled=False,
                temperature=0.3,          # 改写需要一点多样性，不能是 0
                json_mode=True,
            )
            data = parse_llm_json(resp.text or "")
            candidate = str(data.get("query", "")).strip()
            if candidate:
                new_query = candidate
        except Exception as e:
            logger.warning("查询重写失败，保持原查询: %s", e)

        # ── 语义收敛保护 ──
        if _normalize(new_query) == _normalize(last_query):
            logger.info("改写结果与上次相同（%r），强制跳出循环", new_query[:40])
            count = max_rewrites

        logger.info("查询重写 #%d: %r → %r", count, last_query[:30], new_query[:40])

        return usage_update(state, resp, extra={
            "question": new_query,
            "rewrite_count": count,
            "rewrite_history": [new_query],
        })

    return rewrite


def _normalize(text: str) -> str:
    """归一化用于比较查询是否实质变化（去空白与标点差异）"""
    return "".join(ch for ch in (text or "") if ch.isalnum()).lower()


# ═══════════════════════════════════════════════════════════════
# 节点：闲聊直答（不检索）
# ═══════════════════════════════════════════════════════════════

def make_chitchat_node(bundle):
    """闲聊分支：不检索，直接生成简短回应"""

    async def chitchat(state: AgentState) -> dict:
        question = state.get("original_question") or state.get("question", "")
        template = get_agent_prompt("agent_chitchat")
        system, user_msg = template.render(question=question)

        try:
            resp = await bundle.llm.generate(
                system=system,
                user_message=user_msg,
                thinking_enabled=False,
                temperature=0.7,
            )
            answer = resp.text or ""
            ptokens, ctokens = resp.usage.prompt_tokens, resp.usage.completion_tokens
        except Exception as e:
            logger.warning("闲聊生成失败: %s", e)
            answer = "你好，我是基于文档的问答助手。请提出与已上传文档相关的问题。"
            ptokens = ctokens = 0

        return {
            "answer": answer,
            "hits": [],
            "citations": [],
            "refused": False,
            "llm_call_count": (state.get("llm_call_count") or 0) + 1,
            "prompt_tokens": (state.get("prompt_tokens") or 0) + ptokens,
            "completion_tokens": (state.get("completion_tokens") or 0) + ctokens,
        }

    return chitchat
