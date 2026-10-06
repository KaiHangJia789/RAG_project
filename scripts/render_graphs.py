"""
渲染所有 LangGraph 工作流的 mermaid 流程图到文档

用法:
    python scripts/render_graphs.py

产物:
    docs/week11/graph_linear.md      线性流程
    docs/week12/graph_branch.md      条件分支（若已实现）
    docs/week12/graph_loop.md        循环检索（若已实现）
    docs/week13/graph_self_rag.md    Self-RAG 完整链路（若已实现）

**流程图从实际代码生成**，不是手画的 —— 所以永远不会和实现脱节。
改了图结构，重跑本脚本即可。
"""
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.agent.graphs import mermaid_of  # noqa: E402

logging.basicConfig(level=logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]


def render(name: str, builder, out_path: Path, *, title: str, desc: str) -> bool:
    """渲染单个图的 mermaid 并写入文档"""
    try:
        graph = builder()
        compiled = graph if hasattr(graph, "get_graph") else graph.compile()
        mermaid = mermaid_of(compiled)
    except Exception as e:
        print(f"  跳过 {name}: {type(e).__name__}: {e}")
        return False

    if not mermaid:
        print(f"  跳过 {name}: 未能生成 mermaid")
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# {title}

{desc}

> 本图由 `python scripts/render_graphs.py` 从代码自动生成（`graph.get_graph().draw_mermaid()`），
> 不是手绘 —— 改图结构后重跑脚本即可同步。

```mermaid
{mermaid}
```
"""
    out_path.write_text(content, encoding="utf-8")
    print(f"  ✅ {name} → {out_path.relative_to(ROOT)}")
    return True


def main() -> None:
    print("渲染工作流流程图")
    print()

    # 各图用最小的依赖构造 —— 渲染只需要图结构，不需要真实服务
    def linear():
        from app.agent.graphs import build_linear_graph
        from app.agent.nodes import ModelBundle
        return build_linear_graph(ModelBundle()).compile()

    def branch():
        from app.agent.graphs import build_branching_graph
        from app.agent.nodes import ModelBundle
        return build_branching_graph(ModelBundle()).compile()

    def loop():
        from app.agent.graphs import build_self_rag_graph
        from app.agent.nodes import ModelBundle
        return build_self_rag_graph(ModelBundle()).compile()

    def agentic():
        from app.agent.graphs import build_agentic_graph
        from app.agent.nodes import ModelBundle
        return build_agentic_graph(ModelBundle(), None).compile()

    def react():
        from app.agent.graphs import build_react_graph
        from app.agent.nodes import ModelBundle
        from app.agent.tools import build_default_registry
        return build_react_graph(ModelBundle(), build_default_registry()).compile()

    done = 0
    done += render(
        "linear", linear, ROOT / "docs/week11/graph_linear.md",
        title="第 11 周：线性 RAG 工作流",
        desc="接收问题 → 索引检查 → 检索 → 装配上下文 → 生成 → 引用校验。\n"
             "虚线为条件边，用于两条短路路径（索引未就绪 / 检索无命中直接结束，不发 LLM）。",
    )
    done += render(
        "branch", branch, ROOT / "docs/week12/graph_branch.md",
        title="第 12 周：条件分支 RAG",
        desc="按查询类型（事实 / 推理 / 闲聊）路由到不同的检索策略。",
    )
    done += render(
        "loop", loop, ROOT / "docs/week12/graph_loop.md",
        title="第 12 周：循环检索（Self-RAG 骨架）",
        desc="检索 → 相关性评分 → 不达标则重写查询 → 重新检索。\n"
             "回边（rewrite → retrieve）形成循环，由重写次数上限 + 相关性阈值双重终止。",
    )
    done += render(
        "agentic", agentic, ROOT / "docs/week13/graph_agentic.md",
        title="第 13 周：Agentic RAG 完整链路",
        desc="查询分类 →（闲聊直答 | 检索 → 相关性评估 → 重写循环）→ 生成 → 引用。\n"
             "在 Self-RAG 骨架上叠加查询分类分支。",
    )
    done += render(
        "react", react, ROOT / "docs/week13/graph_react.md",
        title="第 13 周：ReAct 工具调用",
        desc="agent（模型自主决定调不调工具）⇄ tools（执行工具）。\n"
             "回边（tools → agent）形成 ReAct 循环，由工具轮次上限终止。\n"
             "finalize 节点用标准引用管线重生成答案（修复 ReAct 模式引用率为 0 的问题）。",
    )

    print()
    print(f"完成：{done} 个图已渲染")


if __name__ == "__main__":
    main()
