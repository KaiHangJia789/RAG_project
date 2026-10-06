# 第 13 周：ReAct 工具调用

agent（模型自主决定调不调工具）⇄ tools（执行工具）。
回边（tools → agent）形成 ReAct 循环，由工具轮次上限终止。
finalize 节点用标准引用管线重生成答案（修复 ReAct 模式引用率为 0 的问题）。

> 本图由 `python scripts/render_graphs.py` 从代码自动生成（`graph.get_graph().draw_mermaid()`），
> 不是手绘 —— 改图结构后重跑脚本即可同步。

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	agent(agent)
	tools(tools)
	finalize(finalize)
	citations(citations)
	__end__([<p>__end__</p>]):::last
	__start__ --> agent;
	agent -. &nbsp;finish&nbsp; .-> finalize;
	agent -.-> tools;
	finalize --> citations;
	tools --> agent;
	citations --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```
