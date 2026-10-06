"""
AgentRunner — Agent 的对外统一入口

**最关键的设计约束：产出 `RagAnswer`，而不是新模型。**

  这样带来三个零成本收益：
    1. `RAGEvaluator.evaluate_one()` 原样可用 —— 不需要写适配层
    2. `scripts/run_rag_eval.py` 加个 `--mode` 参数就能跑 A/B 对比
    3. 前端 `/qa/ask` 的响应格式不变，不用改前端

  如果 Agent 产出自己的结果模型，上面三处都要改，且评测口径可能漂移
  （对比就失去意义了）。
"""
import logging
import time
from typing import Any

from app.config import settings
from app.llm.observability import current_trace_id, trace_url
from app.rag.models import (
    Citation,
    RagAnswer,
    RefusalReason,
    RetrievalConfig,
)
from app.agent.state import initial_state
from app.agent.tracing import AgentTracer, reset_tracer, set_tracer

logger = logging.getLogger("rag_api.agent.runner")


class AgentRunner:
    """
    执行 LangGraph 工作流并把结果转成 RagAnswer。

    用法:
        runner = AgentRunner(compiled_graph)
        answer = await runner.run("什么是 RAG？")
    """

    def __init__(self, compiled_graph: Any, *, mode: str = "agent") -> None:
        self.graph = compiled_graph
        self.mode = mode

    async def run(
        self,
        question: str,
        config: RetrievalConfig | None = None,
        *,
        thread_id: str | None = None,
        max_rewrites: int = 2,
    ) -> RagAnswer:
        """
        执行工作流。

        Args:
            question: 用户问题
            config: 检索/生成参数
            thread_id: 会话 ID（有 checkpointer 时用于多轮状态续接）
            max_rewrites: 查询重写次数上限（循环图的终止条件）
        """
        cfg = config or RetrievalConfig()
        start = time.monotonic()

        state_in = initial_state(
            question, config=cfg.model_dump(), max_rewrites=max_rewrites
        )

        # 有 checkpointer 时必须传 thread_id，否则图无法定位会话状态
        invoke_config: dict = {}
        if thread_id:
            invoke_config["configurable"] = {"thread_id": thread_id}
        # 兜底防无限循环：即使业务层终止条件写错，LangGraph 也会在若干步后中断
        invoke_config["recursion_limit"] = settings.AGENT_RECURSION_LIMIT

        # ── 追踪：根 span 覆盖整次运行，节点 span 显式挂到它下面 ──
        tracer = AgentTracer(
            name=f"agent:{self.mode}",
            metadata={"mode": self.mode, "question": question[:200]},
        )
        token = set_tracer(tracer)
        try:
            with tracer.root(input={"question": question}):
                final_state = await self.graph.ainvoke(state_in, invoke_config)
                # trace_id 必须在 root span **内部**取 —— 退出后 OTel 上下文已失效，
                # current_trace_id() 会返回 None
                trace_id = tracer.trace_id
        finally:
            reset_tracer(token)

        elapsed_ms = (time.monotonic() - start) * 1000

        return self._to_answer(
            final_state, question, cfg, elapsed_ms, trace_id=trace_id
        )

    # ═══════════════════════════════════════════════════════════
    # 状态 → RagAnswer
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _to_answer(
        state: dict,
        question: str,
        cfg: RetrievalConfig,
        elapsed_ms: float,
        *,
        trace_id: str | None = None,
    ) -> RagAnswer:
        """
        把最终状态转成 RagAnswer。

        **字段语义必须与 RagPipeline 保持一致** —— 否则 A/B 对比会因为
        字段填充口径不同而失真（例如 `retrieved_docs` 必须是「检索命中的文档」
        而非「答案引用的文档」，这是 Week10 踩过的坑）。
        """
        hits = state.get("hits") or []

        # 拒答原因字符串 → 枚举（容忍未知值，避免脏数据导致崩溃）
        reason_raw = state.get("refusal_reason") or ""
        reason: RefusalReason | None = None
        if reason_raw:
            try:
                reason = RefusalReason(reason_raw)
            except ValueError:
                logger.warning("未知拒答原因 %r，置空", reason_raw)

        citations = [Citation(**c) for c in (state.get("citations") or [])]

        answer = RagAnswer(
            question=question,
            answer=state.get("answer") or "",
            citations=citations,
            refused=bool(state.get("refused")),
            refusal_reason=reason,
            config=cfg,
            # 与 RagPipeline 同口径：检索命中的文档（去重保序），不是答案引用的
            retrieved_count=len(hits),
            retrieved_docs=list(
                dict.fromkeys(h.get("filename", "") for h in hits)
            ),
            max_score=round(hits[0]["score"], 4) if hits else None,
            latency_ms=elapsed_ms,
            llm_called=(state.get("llm_call_count") or 0) > 0,
            prompt_tokens=state.get("prompt_tokens") or 0,
            completion_tokens=state.get("completion_tokens") or 0,
            # 引用指标直接读 citations 节点在**清洗前**写回的值。
            # 不能在这里重新校验 —— 清洗已删掉越界编号，重校验永远得 1.0，
            # 会让 Agent 与基础 RAG 的同名字段口径不一致（A/B 对比失真）。
            citation_validity=state.get("citation_validity", 1.0),
            invalid_citations=state.get("invalid_citations") or [],
        )

        # 优先用 runner 传入的（在 root span 内取的），回退到当前上下文
        tid = trace_id or current_trace_id()
        if tid:
            answer.trace_id = tid
            answer.trace_url = trace_url(tid)

        return answer
