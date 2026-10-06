"""
评测运行器 — 让「基础 RAG」与「Agentic RAG」走同一套评测流程

**设计要点：两种模式都产出 `RagAnswer`。**

  这是让 A/B 对比成立的关键 —— `RAGEvaluator.evaluate_one(result: RagAnswer)`
  不需要任何适配层，两条路径的指标口径天然一致。

  如果 Agent 产出自己的结果模型，就要写适配层，而适配层正是口径漂移的温床
  （Week10 就踩过 `retrieved_docs` 语义不一致的坑）。
"""
import logging
from typing import Any

from app.config import settings
from app.rag.models import RagAnswer, RetrievalConfig

logger = logging.getLogger("rag_api.eval.runners")

# 运行模式
MODE_BASIC = "basic"       # 命令式管线（RagPipeline）
MODE_AGENT = "agent"       # 图式编排（AgentRunner + agentic 图）
MODE_REACT = "react"       # ReAct 工具调用图


class BaseRunner:
    """基础 RAG：直接调 RagPipeline"""

    mode = MODE_BASIC

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    async def run(self, question: str, config: RetrievalConfig, **kwargs) -> RagAnswer:
        return await self.pipeline.answer(question, config)

    def describe(self) -> str:
        return "基础 RAG（RagPipeline 命令式编排）"


class AgentRunnerAdapter:
    """
    Agent 模式：调 AgentRunner。

    命名带 Adapter 是为了与 `app/agent/runner.py` 的 `AgentRunner` 区分 ——
    那个是图执行器，这个是评测层的统一接口。
    """

    def __init__(self, runner, mode: str = MODE_AGENT) -> None:
        self.runner = runner
        self.mode = mode

    async def run(self, question: str, config: RetrievalConfig, **kwargs) -> RagAnswer:
        return await self.runner.run(question, config)

    def describe(self) -> str:
        return f"Agentic RAG（{self.mode} 图式编排）"


def build_runner(mode: str, index_service, *, db=None, llm=None) -> Any:
    """
    按模式装配运行器。

    Args:
        mode: basic / agent / react
        index_service: 检索服务
        db: 数据库（Agent 模式注册 query_metadata 工具用）
        llm: LLM 客户端（测试注入替身用）
    """
    from app.llm.client import LLMClient

    llm = llm or LLMClient()

    if mode == MODE_BASIC:
        from app.rag.pipeline import RagPipeline
        return BaseRunner(RagPipeline(index_service=index_service, llm=llm))

    from app.agent.runner import AgentRunner
    from app.agent.graphs import (
        build_agentic_graph,
        build_react_graph,
        compile_graph,
    )
    from app.agent.nodes import ModelBundle
    from app.agent.tools import build_default_registry

    bundle = ModelBundle(index_service=index_service, llm=llm)

    if mode == MODE_AGENT:
        graph = compile_graph(build_agentic_graph(bundle, None))
        return AgentRunnerAdapter(AgentRunner(graph, mode=MODE_AGENT), MODE_AGENT)

    if mode == MODE_REACT:
        registry = build_default_registry(index_service=index_service, db=db)
        graph = compile_graph(build_react_graph(bundle, registry))
        return AgentRunnerAdapter(AgentRunner(graph, mode=MODE_REACT), MODE_REACT)

    raise ValueError(f"未知模式 '{mode}'，可用: {MODE_BASIC} / {MODE_AGENT} / {MODE_REACT}")
