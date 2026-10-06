"""
Agent 节点 — 把 app/rag/steps.py 的原子步骤包装成 LangGraph 节点

**节点约定**（LangGraph 的规矩）：
  - 签名 `(state) -> dict`：接收完整 state，返回**部分更新**
  - 不修改传入的 state（要当成只读）
  - 累加型字段（messages/tool_results/rewrite_history）由 State 的 reducer 处理，
    节点只管返回新元素即可

**与 RagPipeline 的关系**：
  两者调用**同一批 steps.py 函数**。差异只在编排：
    - RagPipeline：`retrieve → generate → attach_citations` 顺序写死
    - Agent：同样三步，但中间可插入分类、评分、重写、工具调用，且能回边循环

**节点通过 `ModelBundle` 拿到依赖**（索引服务、LLM），而不是闭包捕获 ——
闭包会让节点难以单测，而显式传依赖可以像测试 RagPipeline 那样注入替身。
"""
import json
import logging
from dataclasses import dataclass

from app.config import settings
from app.llm.client import LLMClient
from app.llm.prompts import PROMPT_REGISTRY
from app.rag import steps
from app.rag.models import RefusalReason, RetrievalConfig
from app.agent.serialization import hits_to_dicts
from app.agent.state import AgentState

logger = logging.getLogger("rag_api.agent.nodes")


@dataclass
class ModelBundle:
    """
    节点需要的依赖集合。

    打包成一个对象传，而不是每个节点单独收参数 —— 这样新增依赖时
    不用改所有节点签名，也方便测试时整体替换。
    """

    index_service: object = None
    llm: LLMClient | None = None


# ═══════════════════════════════════════════════════════════════
# 配置辅助
# ═══════════════════════════════════════════════════════════════

def config_of(state: AgentState) -> RetrievalConfig:
    """
    从 state 还原 RetrievalConfig。

    state 里存 dict 是因为 checkpointer 要可序列化；这里还原成模型
    以便复用 steps.py 的既有逻辑（它们都吃 RetrievalConfig）。
    """
    raw = state.get("config") or {}
    if not raw:
        return RetrievalConfig()
    # 过滤掉未知键，避免配置演进后旧 checkpoint 反序列化失败
    known = set(RetrievalConfig.model_fields)
    return RetrievalConfig(**{k: v for k, v in raw.items() if k in known})


# ═══════════════════════════════════════════════════════════════
# 节点 1：准备（索引就绪检查）
# ═══════════════════════════════════════════════════════════════

def make_prepare_node(bundle: ModelBundle):
    """索引就绪检查。不就绪时直接给出拒答，后续节点由条件边短路跳过。"""

    def prepare(state: AgentState) -> dict:
        if not steps.is_index_ready(bundle.index_service):
            return {
                "refused": True,
                "refusal_reason": RefusalReason.INDEX_NOT_READY.value,
                "answer": "知识库尚未就绪，请先上传文档或构建索引。",
            }
        return {"refused": False, "refusal_reason": ""}

    return prepare


# ═══════════════════════════════════════════════════════════════
# 节点 2：检索
# ═══════════════════════════════════════════════════════════════

def make_retrieve_node(bundle: ModelBundle):
    """
    检索节点。

    用 state 里的 `question`（可能是重写后的）而不是原始问题 ——
    这正是循环重写能生效的关键。
    """

    async def retrieve(state: AgentState) -> dict:
        query = state.get("question") or state.get("original_question", "")
        cfg = config_of(state)

        hits = await steps.retrieve(bundle.index_service, query, cfg)

        update: dict = {"hits": hits_to_dicts(hits)}

        if not hits:
            # 检索无命中 → 阈值层拒答。是否重写由条件边决定，这里只报告事实。
            update["refused"] = True
            update["refusal_reason"] = RefusalReason.BELOW_THRESHOLD.value
            logger.info("检索无命中: q=%r (min_score=%s)", query[:40], cfg.min_score)
        else:
            update["refused"] = False
            update["refusal_reason"] = ""

        return update

    return retrieve


# ═══════════════════════════════════════════════════════════════
# 节点 3：装配上下文
# ═══════════════════════════════════════════════════════════════

def make_assemble_node(bundle: ModelBundle):
    """把检索结果拼成带 [n] 编号的上下文"""

    def assemble(state: AgentState) -> dict:
        hits = state.get("hits") or []
        if not hits:
            return {"context": ""}
        from app.agent.serialization import dicts_to_hits

        return {"context": steps.assemble_context(dicts_to_hits(hits))}

    return assemble


# ═══════════════════════════════════════════════════════════════
# 节点 4：生成
# ═══════════════════════════════════════════════════════════════

def make_generate_node(bundle: ModelBundle):
    """
    生成答案。

    这里刻意**不**做引用校验（那是下一个节点的职责）——
    保持节点职责单一，这样引用校验节点可以被单独复用/替换。
    """

    async def generate(state: AgentState) -> dict:
        cfg = config_of(state)
        context = state.get("context") or ""
        question = state.get("original_question") or state.get("question", "")

        resp = await steps.generate_answer(
            bundle.llm, question, context, config=cfg
        )

        return {
            "answer": resp.text or "",
            "llm_call_count": (state.get("llm_call_count") or 0) + 1,
            "prompt_tokens": (state.get("prompt_tokens") or 0) + resp.usage.prompt_tokens,
            "completion_tokens": (
                (state.get("completion_tokens") or 0) + resp.usage.completion_tokens
            ),
        }

    return generate


# ═══════════════════════════════════════════════════════════════
# 节点 5：引用校验
# ═══════════════════════════════════════════════════════════════

def make_citation_node(bundle: ModelBundle):
    """
    校验 [n] 引用编号并映射回检索结果。

    **必须把 validity / invalid 写回 state**，不能只返回清洗后的答案 ——
    因为清洗把越界编号删掉了，下游若再校验一次就永远查不到问题
    （这是 A/B 对比最容易失真的地方：两边字段填充口径不一致）。
    """

    def attach(state: AgentState) -> dict:
        from app.agent.serialization import dicts_to_hits

        hits = dicts_to_hits(state.get("hits") or [])
        result = steps.attach_citations(state.get("answer") or "", hits)

        update: dict = {
            "answer": result.text,
            "citations": [c.model_dump() for c in result.citations],
            "citation_validity": result.validity,
            "invalid_citations": result.invalid,
        }
        if result.is_refusal:
            update["refused"] = True
            update["refusal_reason"] = RefusalReason.LLM_REFUSED.value

        return update

    return attach
