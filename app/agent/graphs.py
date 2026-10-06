"""
LangGraph 工作流构建

三个图对应三周的产出：
  - build_linear_graph()    第 11 周：接收问题 → 检索 → 回答（线性）
  - build_branching_graph() 第 12 周：按查询类型走不同检索策略
  - build_self_rag_graph()  第 12/13 周：检索 → 评分 → 不达标则重写重检（循环）
                            并支持工具调用（第 13 周）

**图是可编译的**：`build_*` 返回未编译的 `StateGraph` 或已编译图。
编译时必须传入 checkpointer 才能支持多轮会话（第 12 周内容）。

**可视化**：编译后调 `.get_graph().draw_mermaid()` 可直接产出 mermaid 文本，
渲染出来就是验收要求的「完整 Agent 流程图」。
"""
import logging

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    ModelBundle,
    make_assemble_node,
    make_citation_node,
    make_generate_node,
    make_prepare_node,
    make_retrieve_node,
)
from app.agent.state import AgentState
from app.agent.tracing import traced_node

logger = logging.getLogger("rag_api.agent.graphs")

# 节点名 → LangFuse span 类型（语义化类型让 Dashboard 里一眼看出节点在干什么）
_NODE_SPAN_TYPES = {
    "retrieve": "retriever",
    "generate": "generation",
    "classify": "chain",
    "grade": "evaluator",
    "rewrite": "chain",
    "chitchat": "generation",
    "citations": "chain",
    "assemble": "chain",
    "prepare": "chain",
    "tools": "tool",
}


def _node(name: str, fn):
    """给节点函数套上追踪 span（无 LangFuse 时自动降级为直通）"""
    return traced_node(name, fn, as_type=_NODE_SPAN_TYPES.get(name, "span"))


# ═══════════════════════════════════════════════════════════════
# 第 11 周：线性图
# ═══════════════════════════════════════════════════════════════

def build_linear_graph(bundle: ModelBundle):
    """
    线性流程：prepare → retrieve → assemble → generate → citations

    与 `RagPipeline.answer()` 的步骤**完全对应**，但被拆成了显式的节点与边。
    对比两者代码结构，就能看出 LangGraph 把「顺序」变成了可声明、可观测、
    可插入新环节的对象。

    条件边 `route_after_prepare` / `route_after_retrieve` 负责短路：
    索引未就绪或检索无命中时直接到 END，不浪费时间走后续节点。
    （线性图里也有条件边，因为拒答短路本质上就是分支。）
    """
    graph = StateGraph(AgentState)

    graph.add_node("prepare", _node("prepare", make_prepare_node(bundle)))
    graph.add_node("retrieve", _node("retrieve", make_retrieve_node(bundle)))
    graph.add_node("assemble", _node("assemble", make_assemble_node(bundle)))
    graph.add_node("generate", _node("generate", make_generate_node(bundle)))
    graph.add_node("citations", _node("citations", make_citation_node(bundle)))

    graph.add_edge(START, "prepare")

    # 索引未就绪 → 直接结束
    graph.add_conditional_edges(
        "prepare",
        _route_after_prepare,
        {"continue": "retrieve", "end": END},
    )

    # 检索无命中 → 直接结束（阈值层拒答，不发 LLM）
    graph.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {"continue": "assemble", "end": END},
    )

    graph.add_edge("assemble", "generate")
    graph.add_edge("generate", "citations")
    graph.add_edge("citations", END)

    return graph


# ═══════════════════════════════════════════════════════════════
# 路由函数（条件边的判定逻辑）
# ═══════════════════════════════════════════════════════════════

def _route_after_prepare(state: AgentState) -> str:
    """索引未就绪 → 结束；否则继续检索"""
    if state.get("refused") and state.get("refusal_reason") == "index_not_ready":
        return "end"
    return "continue"


def _route_after_retrieve(state: AgentState) -> str:
    """检索无命中 → 结束（拒答）；否则继续生成"""
    hits = state.get("hits") or []
    return "continue" if hits else "end"


# ═══════════════════════════════════════════════════════════════
# 第 12 周：条件分支图（按查询类型路由）
# ═══════════════════════════════════════════════════════════════

def build_branching_graph(bundle: ModelBundle):
    """
    条件分支：classify → 按类型走不同路径

    - chitchat  → 直答（不检索），省掉一次无意义的检索
    - factual   → 标准检索参数（final_k 较小，事实通常集中在一处）
    - reasoning → 扩大检索范围（final_k 更大，答案分散在多个段落）

    分支参数通过节点内改写 config 实现，而不是建三条重复的子图 ——
    后者会让流程图爆炸且难维护。
    """
    from app.agent.nodes_routing import make_chitchat_node, make_classify_node

    graph = StateGraph(AgentState)

    graph.add_node("prepare", _node("prepare", make_prepare_node(bundle)))
    graph.add_node("classify", _node("classify", make_classify_node(bundle)))
    graph.add_node("chitchat", _node("chitchat", make_chitchat_node(bundle)))
    graph.add_node("retrieve", _node("retrieve", make_retrieve_node(bundle)))
    graph.add_node("assemble", _node("assemble", make_assemble_node(bundle)))
    graph.add_node("generate", _node("generate", make_generate_node(bundle)))
    graph.add_node("citations", _node("citations", make_citation_node(bundle)))

    graph.add_edge(START, "prepare")
    graph.add_conditional_edges(
        "prepare", _route_after_prepare,
        {"continue": "classify", "end": END},
    )
    graph.add_conditional_edges(
        "classify", _route_after_classify,
        {"chitchat": "chitchat", "retrieve": "retrieve"},
    )

    graph.add_edge("chitchat", END)
    graph.add_conditional_edges(
        "retrieve", _route_after_retrieve,
        {"continue": "assemble", "end": END},
    )
    graph.add_edge("assemble", "generate")
    graph.add_edge("generate", "citations")
    graph.add_edge("citations", END)

    return graph


def _route_after_classify(state: AgentState) -> str:
    """闲聊直接答，其余走检索"""
    return "chitchat" if state.get("query_type") == "chitchat" else "retrieve"


# ═══════════════════════════════════════════════════════════════
# 第 12 周：循环检索图（Self-RAG 骨架）
# ═══════════════════════════════════════════════════════════════

def build_self_rag_graph(bundle: ModelBundle):
    """
    循环检索：retrieve → assemble → grade → (rewrite → retrieve)* → generate

    这是第 12 周「循环节点」与第 13 周「Self-RAG」共用的骨架。
    回边 `rewrite → retrieve` 形成循环，由**三重终止条件**保护：

      1. 业务上限  `rewrite_count >= max_rewrites`（默认 2）
      2. 语义收敛  改写结果与上次相同 → 重写节点强制把 count 推到上限
      3. 平台兜底  invoke 时的 `recursion_limit`（runner 里设 25）

    缺任何一重都可能出现「检索→评分不变→改写不变→再检索」的空转。
    """
    from app.agent.nodes_routing import (
        make_grade_node,
        make_rewrite_node,
    )

    graph = StateGraph(AgentState)

    graph.add_node("prepare", _node("prepare", make_prepare_node(bundle)))
    graph.add_node("retrieve", _node("retrieve", make_retrieve_node(bundle)))
    graph.add_node("assemble", _node("assemble", make_assemble_node(bundle)))
    graph.add_node("grade", _node("grade", make_grade_node(bundle)))
    graph.add_node("rewrite", _node("rewrite", make_rewrite_node(bundle)))
    graph.add_node("generate", _node("generate", make_generate_node(bundle)))
    graph.add_node("citations", _node("citations", make_citation_node(bundle)))

    graph.add_edge(START, "prepare")
    graph.add_conditional_edges(
        "prepare", _route_after_prepare,
        {"continue": "retrieve", "end": END},
    )

    # 检索无命中 → 不直接结束，交给 grade 决定是否重写
    graph.add_edge("retrieve", "assemble")
    graph.add_edge("assemble", "grade")

    graph.add_conditional_edges(
        "grade", _route_after_grade,
        {"generate": "generate", "rewrite": "rewrite"},
    )

    # 回边：改写后重新检索（循环）
    graph.add_edge("rewrite", "retrieve")

    graph.add_edge("generate", "citations")
    graph.add_edge("citations", END)

    return graph


def _route_after_grade(state: AgentState) -> str:
    """
    相关性达标 → 生成；不达标且还有重写额度 → 重写。

    阈值取自 state 的 config（便于 A/B 调参），默认 0.5。
    注意**不复用** RAG_MIN_SCORE —— 那个是 FAISS 余弦分，与这里的判官
    语义评分量纲不同，混用会导致「以为在调检索阈值，实际在调判官阈值」。
    """
    from app.config import settings

    score = state.get("relevance_score")
    score = 1.0 if score is None else float(score)
    threshold = settings.AGENT_RELEVANCE_THRESHOLD

    if score >= threshold:
        return "generate"

    count = state.get("rewrite_count") or 0
    max_rewrites = state.get("max_rewrites", settings.AGENT_MAX_REWRITES)
    if count >= max_rewrites:
        # 额度用尽 → 带着当前上下文生成（由 prompt 决定是否拒答）
        return "generate"

    return "rewrite"


# ═══════════════════════════════════════════════════════════════
# 第 13 周：Agentic RAG（Self-RAG + 工具调用）
# ═══════════════════════════════════════════════════════════════

def build_agentic_graph(bundle: ModelBundle, registry):
    """
    Agentic RAG 完整链路：

        prepare → classify → [chitchat 直答 | 检索]
        → assemble → grade → [达标生成 | 不达标重写并重新检索]
        → generate → citations → END

    在 self_rag 的基础上叠加两处 Agent 能力：
      1. **查询分类**（第 12 周）：闲聊走直答，跳过检索
      2. **相关性评估 + 重写循环**（第 12 周）：检索质量不达标时自动重试

    工具调用由 `build_react_graph` 单独承载 —— 把它和检索循环混在一张图里
    会让流程难以调试，且两者的失败模式不同（工具失败 vs 检索质量差）。
    """
    from app.agent.nodes_routing import (
        make_chitchat_node,
        make_classify_node,
        make_grade_node,
        make_rewrite_node,
    )

    graph = StateGraph(AgentState)

    graph.add_node("prepare", _node("prepare", make_prepare_node(bundle)))
    graph.add_node("classify", _node("classify", make_classify_node(bundle)))
    graph.add_node("chitchat", _node("chitchat", make_chitchat_node(bundle)))
    graph.add_node("retrieve", _node("retrieve", make_retrieve_node(bundle)))
    graph.add_node("assemble", _node("assemble", make_assemble_node(bundle)))
    graph.add_node("grade", _node("grade", make_grade_node(bundle)))
    graph.add_node("rewrite", _node("rewrite", make_rewrite_node(bundle)))
    graph.add_node("generate", _node("generate", make_generate_node(bundle)))
    graph.add_node("citations", _node("citations", make_citation_node(bundle)))

    graph.add_edge(START, "prepare")
    graph.add_conditional_edges(
        "prepare", _route_after_prepare, {"continue": "classify", "end": END},
    )
    graph.add_conditional_edges(
        "classify", _route_after_classify,
        {"chitchat": "chitchat", "retrieve": "retrieve"},
    )

    graph.add_edge("chitchat", END)
    graph.add_edge("retrieve", "assemble")
    graph.add_edge("assemble", "grade")
    graph.add_conditional_edges(
        "grade", _route_after_grade, {"generate": "generate", "rewrite": "rewrite"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", "citations")
    graph.add_edge("citations", END)

    return graph


def build_react_graph(bundle: ModelBundle, registry):
    """
    ReAct 工具调用图：

        agent → [有工具调用？] → tools → agent → ... → END

    模型自主决定调哪个工具、调几次。轮次上限 `AGENT_TOOL_MAX_ROUNDS` 是
    硬约束（在 `_route_after_agent` 里检查）。

    与 `build_agentic_graph` 分开的原因：两者的失败模式与调试方式不同。
    检索循环看的是"相关性评分曲线"，工具循环看的是"工具调用日志"。
    合并成一张图会让这两条线索互相淹没。
    """
    from app.agent.nodes_tools import (
        _route_after_agent,
        make_finalize_node,
        make_tool_agent_node,
        make_tool_executor_node,
    )

    graph = StateGraph(AgentState)

    graph.add_node("agent", _node("agent", make_tool_agent_node(bundle, registry)))
    graph.add_node("tools", _node("tools", make_tool_executor_node(bundle, registry)))
    graph.add_node("finalize", _node("finalize", make_finalize_node(bundle)))
    graph.add_node("citations", _node("citations", make_citation_node(bundle)))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", _route_after_agent, {"tools": "tools", "finish": "finalize"},
    )
    # 回边：工具执行完回到 agent 继续决策（ReAct 循环）
    graph.add_edge("tools", "agent")
    # 收尾：用标准引用管线重生成答案（有检索命中且原答案无引用时才真正执行）
    graph.add_edge("finalize", "citations")
    graph.add_edge("citations", END)

    return graph


# ═══════════════════════════════════════════════════════════════
# 编译辅助
# ═══════════════════════════════════════════════════════════════

def compile_graph(graph, *, checkpointer=None):
    """
    编译图。

    Args:
        checkpointer: 第 12 周的 SQLite checkpointer；None = 无状态（单次执行）
    """
    return graph.compile(checkpointer=checkpointer)


def mermaid_of(compiled_graph) -> str:
    """
    产出 mermaid 流程图文本。

    这是验收产出「用 mermaid 画出完整的 Agent 流程图」的直接来源 ——
    图是从**实际代码**生成的，不是手画的，所以永远不会和实现脱节。
    """
    try:
        return compiled_graph.get_graph().draw_mermaid()
    except Exception as e:
        logger.warning("生成 mermaid 失败: %s", e)
        return ""
