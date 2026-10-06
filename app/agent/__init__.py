"""
Agent 编排层 — LangGraph 工作流

与 `app/rag/` 的关系:
  - `app/rag/steps.py`   原子步骤（两边共用）
  - `app/rag/pipeline.py` 命令式编排（顺序写死）
  - `app/agent/`          图式编排（条件分支 + 循环 + 工具调用）

  对比这三者，就是「普通 RAG vs LangGraph」的代码结构差异本身。

注意:
  - State 里只放可序列化的 dict（checkpointer 要落盘）
  - 消息列表用 `operator.add` 而非 `add_messages` —— 后者是 LangChain 专用
    reducer，会丢 `reasoning_content`，而 DeepSeek 在带 tools 时要求原样回传
"""
from app.agent.nodes import ModelBundle
from app.agent.runner import AgentRunner
from app.agent.state import AgentState, initial_state

__all__ = [
    "AgentRunner",
    "AgentState",
    "ModelBundle",
    "initial_state",
]
