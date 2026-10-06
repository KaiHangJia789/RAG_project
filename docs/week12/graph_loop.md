# 第 12 周：循环检索（Self-RAG 骨架）

检索 → 相关性评分 → 不达标则重写查询 → 重新检索。
回边（rewrite → retrieve）形成循环，由重写次数上限 + 相关性阈值双重终止。

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
	prepare(prepare)
	retrieve(retrieve)
	assemble(assemble)
	grade(grade)
	rewrite(rewrite)
	generate(generate)
	citations(citations)
	__end__([<p>__end__</p>]):::last
	__start__ --> prepare;
	assemble --> grade;
	generate --> citations;
	grade -.-> generate;
	grade -.-> rewrite;
	prepare -. &nbsp;end&nbsp; .-> __end__;
	prepare -. &nbsp;continue&nbsp; .-> retrieve;
	retrieve --> assemble;
	rewrite --> retrieve;
	citations --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```
