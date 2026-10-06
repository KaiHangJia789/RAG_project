"""
Node 级追踪 — 让 LangGraph 的每个节点在 LangFuse 里成为一个 span

**要解决的核心问题**：
  LangGraph 的节点在各自的 asyncio 任务里执行（Pregel 的 superstep 调度）。
  LangFuse 4.x 基于 OpenTelemetry contextvars，而 contextvar 在
  `asyncio.create_task` 时会被复制 —— 兄弟节点**未必**自动归到同一 trace 下。

  **解法：不依赖 contextvar 继承，用显式 trace_context 挂父节点。**
  `start_as_current_observation(trace_context={"trace_id": ..., "parent_span_id": ...})`
  是确定性方案，不依赖调度实现细节。

用法：
    tracer = AgentTracer()
    with tracer.span("agent:ask", as_type="agent") as root:
        tid, sid = tracer.current_ids()
        with tracer.span("retrieve", as_type="retriever", parent=(tid, sid)):
            ...

  没有配置 LangFuse 时全部降级为 no-op，不影响功能。
"""
import contextvars
import functools
import inspect
import logging
from contextlib import contextmanager
from typing import Any, Callable

from app.llm.observability import get_langfuse, langfuse_enabled

logger = logging.getLogger("rag_api.agent.tracing")

# 当前运行的 tracer。用 contextvar 而非全局变量，因为并发的问答请求
# 各有自己的 trace，全局变量会串线。
_current_tracer: contextvars.ContextVar["AgentTracer | None"] = (
    contextvars.ContextVar("agent_tracer", default=None)
)


def set_tracer(tracer: "AgentTracer | None") -> contextvars.Token:
    """设置当前 tracer（返回 token 供恢复）"""
    return _current_tracer.set(tracer)


def reset_tracer(token: contextvars.Token) -> None:
    _current_tracer.reset(token)


def get_tracer() -> "AgentTracer | None":
    return _current_tracer.get()


def traced_node(node_name: str, fn: Callable, *, as_type: str = "span") -> Callable:
    """
    把节点函数包一层 span（支持同步与异步节点）。

    **父子关系用显式传参**（从 tracer 拿根 span 的 id），不依赖 contextvar
    跨 asyncio 任务的继承 —— LangGraph 的 Pregel 调度会把节点放进各自的 task，
    而 contextvar 在 `create_task` 时是复制而非共享，兄弟节点能否归到同一
    trace 取决于框架实现细节，不可靠。

    追踪失败绝不影响节点执行（span 创建已被 `AgentTracer.span` 内部 try 包裹）。
    """

    def _wrap_sync(state):
        tracer = get_tracer()
        if tracer is None or not tracer.enabled:
            return fn(state)
        with tracer.span(
            f"node:{node_name}", as_type=as_type,
            input={"question": state.get("question", "")[:100]},
            parent=tracer.current_ids(),
        ) as span:
            out = fn(state)
            if span is not None and isinstance(out, dict):
                _record(span, out)
            return out

    async def _wrap_async(state):
        tracer = get_tracer()
        if tracer is None or not tracer.enabled:
            return await fn(state)
        with tracer.span(
            f"node:{node_name}", as_type=as_type,
            input={"question": state.get("question", "")[:100]},
            parent=tracer.current_ids(),
        ) as span:
            out = await fn(state)
            if span is not None and isinstance(out, dict):
                _record(span, out)
            return out

    wrapper = _wrap_async if inspect.iscoroutinefunction(fn) else _wrap_sync
    return functools.wraps(fn)(wrapper)


def _record(span, out: dict) -> None:
    """把节点的输出摘要记进 span（只记关键字段，避免状态太大）"""
    try:
        summary = {}
        for k in ("query_type", "relevance_score", "rewrite_count",
                  "refused", "refusal_reason", "llm_call_count"):
            if k in out:
                summary[k] = out[k]
        if "hits" in out:
            summary["hits"] = len(out["hits"] or [])
        if "answer" in out:
            summary["answer_preview"] = str(out["answer"])[:150]
        span.update(output=summary or None)
    except Exception as e:
        logger.debug("记录 span 输出失败: %s", e)


class AgentTracer:
    """
    节点级 span 管理器。

    持有根 span 的 (trace_id, span_id)，供各节点显式挂父。
    """

    def __init__(self, name: str = "agent-run", metadata: dict | None = None) -> None:
        self.name = name
        self.metadata = metadata or {}
        self.trace_id: str | None = None
        self.root_span_id: str | None = None
        self._client = get_langfuse() if langfuse_enabled() else None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @contextmanager
    def root(self, *, input: Any = None, metadata: dict | None = None):
        """
        开根 span（整个工作流一个 trace）。

        Yields:
            根 span 对象（未启用时为 None）
        """
        if not self.enabled:
            yield None
            return

        merged = {**self.metadata, **(metadata or {})}
        with self._client.start_as_current_observation(
            name=self.name,
            as_type="agent",
            input=input,
            metadata=merged or None,
        ) as span:
            # 记下 id，供子节点显式挂父
            self.trace_id = self._client.get_current_trace_id()
            self.root_span_id = self._client.get_current_observation_id()
            try:
                yield span
            finally:
                pass

    @contextmanager
    def span(
        self,
        name: str,
        *,
        as_type: str = "span",
        input: Any = None,
        metadata: dict | None = None,
        parent: tuple[str | None, str | None] | None = None,
    ):
        """
        开一个节点 span。

        Args:
            as_type: span 类型 —— LangFuse 支持 span/agent/tool/chain/retriever/
                generation/embedding/evaluator/guardrail。用语义化的类型能让
                Dashboard 里一眼看出哪个节点在干什么。
            parent: (trace_id, parent_span_id)。显式传入可保证嵌套正确，
                不传则依赖 contextvar 继承（在 LangGraph 里不可靠）。
        """
        if not self.enabled:
            yield None
            return

        ctx = None
        if parent is not None:
            trace_id, parent_id = parent
            if trace_id:
                ctx = {"trace_id": trace_id}
                if parent_id:
                    ctx["parent_span_id"] = parent_id

        try:
            with self._client.start_as_current_observation(
                name=name,
                as_type=as_type,
                input=input,
                metadata=metadata or None,
                trace_context=ctx,
            ) as span:
                yield span
        except Exception as e:
            # 追踪失败绝不能影响主流程
            logger.warning("创建 span '%s' 失败: %s", name, e)
            yield None

    def current_ids(self) -> tuple[str | None, str | None]:
        """当前 (trace_id, span_id)，供传给子节点当 parent"""
        return self.trace_id, self.root_span_id

    def score(self, name: str, value: float, *, comment: str | None = None) -> None:
        """给当前 trace 打分（用于把 Agent 的过程指标也上报）"""
        if not self.enabled:
            return
        try:
            self._client.score_current_trace(name=name, value=value, comment=comment)
        except Exception as e:
            logger.debug("打分 %s 失败: %s", name, e)


@contextmanager
def node_span(name: str, *, as_type: str = "span", input: Any = None):
    """
    轻量级节点 span（不显式指定父节点，依赖 contextvar 继承）。

    适用于**单节点独立调用**（如单测），不适用于 LangGraph 图内 ——
    图内请用 `AgentTracer.span(..., parent=...)`。
    """
    if not langfuse_enabled():
        yield None
        return
    client = get_langfuse()
    if client is None:
        yield None
        return
    try:
        with client.start_as_current_observation(
            name=name, as_type=as_type, input=input
        ) as span:
            yield span
    except Exception as e:
        logger.warning("创建 span '%s' 失败: %s", name, e)
        yield None
