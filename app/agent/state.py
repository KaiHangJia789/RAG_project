"""
AgentState — LangGraph 工作流的共享状态

**设计原则**：字段随三周需求递增，但一次性定好骨架。

  - 第 11 周只需 question / hits / answer / citations（线性流程）
  - 第 12 周加 query_type / rewrite_count / relevance_score（分支与循环）
  - 第 13 周加 tool_calls / tool_results / messages（工具调用）

  一次性定义的好处：后续周只加节点，不动 State 结构，避免每次都要改所有节点签名。

**Reducer 选择**：
  - 消息列表用 `add_messages` 语义（累加）—— 但项目不用 LangChain 的消息类，
    所以用 `operator.add` 累加普通 dict 列表
  - 其余字段默认**覆盖写**（last-write-wins），这是 LangGraph 的默认行为

**为什么用 TypedDict 而不是 Pydantic**：
  LangGraph 的 State 需要是 TypedDict（或 dataclass），Pydantic 模型会限制
  reducer 的声明方式。用 TypedDict 是最贴合框架的写法。

用法提示：节点函数返回**部分状态**（只含要更新的字段），不需要返回完整 state。
"""
import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    """Agent 工作流的共享状态（total=False 允许部分字段存在）"""

    # ═══════════════════════════════════════════════════════════
    # 输入
    # ═══════════════════════════════════════════════════════════

    question: str
    """用户原始问题"""

    original_question: str
    """改写前的原始问题（循环重写时保留，用于日志与调试）"""

    config: dict
    """检索配置（RetrievalConfig.model_dump()）。用 dict 而非模型实例，
    因为 LangGraph 的 checkpointer 需要可序列化的状态。"""

    # ═══════════════════════════════════════════════════════════
    # 第 11 周：线性流程
    # ═══════════════════════════════════════════════════════════

    hits: list[dict]
    """检索命中的 chunk（IndexHit 序列化后的 dict）"""

    context: str
    """装配好的带编号上下文"""

    answer: str
    """最终答案"""

    citations: list[dict]
    """引用来源（Citation 序列化后的 dict）"""

    citation_validity: float
    """引用编号有效率（由 citations 节点在**清洗前**算出并写回）。

    不能在下游重新校验 —— 清洗会把越界编号删掉，重校验永远得到 1.0，
    导致 A/B 两边的 citation_validity 口径不一致。
    """

    invalid_citations: list[int]
    """被剔除的越界引用编号（同上，必须在清洗前记录）"""

    # ═══════════════════════════════════════════════════════════
    # 第 12 周：条件分支与循环
    # ═══════════════════════════════════════════════════════════

    query_type: str
    """查询分类：factual（事实查询）/ reasoning（推理查询）/ chitchat（闲聊）"""

    relevance_score: float
    """检索结果对问题的相关性评分（0-1），低于阈值触发重写"""

    relevance_missing: str
    """评分节点指出的「缺少什么信息」，作为重写节点的输入"""

    rewrite_count: int
    """查询重写次数（循环终止条件的依据之一）"""

    max_rewrites: int
    """重写次数上限（防止无限循环）"""

    rewrite_history: Annotated[list[str], operator.add]
    """历次改写后的查询（累加，便于调试循环过程）"""

    # ═══════════════════════════════════════════════════════════
    # 第 13 周：工具调用
    # ═══════════════════════════════════════════════════════════

    messages: Annotated[list[dict], operator.add]
    """完整的消息流（OpenAI 格式），累加。

    注意：assistant 消息必须保留 `reasoning_content` —— DeepSeek 在带 tools 的
    请求里要求原样回传，否则 HTTP 400（见 scripts/spike_tool_calling.py 的实测）。
    """

    tool_calls: list[dict]
    """本轮模型请求的工具调用"""

    tool_results: Annotated[list[dict], operator.add]
    """工具执行结果（累加，保留完整调用日志）"""

    tool_round: int
    """工具调用轮次（防止 Agent 无限调工具）"""

    # ═══════════════════════════════════════════════════════════
    # 输出与诊断
    # ═══════════════════════════════════════════════════════════

    refused: bool
    """是否拒答"""

    refusal_reason: str
    """拒答原因（RefusalReason 的值）"""

    error: str
    """执行过程中的错误"""

    llm_call_count: int
    """LLM 调用次数（A/B 对比成本用）"""

    prompt_tokens: int
    """累计输入 token"""

    completion_tokens: int
    """累计输出 token"""

    trace_id: str
    """LangFuse trace ID"""


def initial_state(
    question: str,
    *,
    config: dict | None = None,
    max_rewrites: int = 2,
) -> AgentState:
    """
    构造初始状态。

    集中在这里构造的好处：所有默认值一处可见，新增字段时不会漏初始化。
    """
    return AgentState(
        question=question,
        original_question=question,
        config=config or {},
        max_rewrites=max_rewrites,
        rewrite_count=0,
        tool_round=0,
        llm_call_count=0,
        prompt_tokens=0,
        completion_tokens=0,
        refused=False,
        rewrite_history=[],
        tool_results=[],
    )
