# 第 13 周：Agentic RAG 完整链路

查询分类 →（闲聊直答 | 检索 → 相关性评估 → 重写循环）→ 生成 → 引用。
在 Self-RAG 骨架上叠加查询分类分支。

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
	classify(classify)
	chitchat(chitchat)
	retrieve(retrieve)
	assemble(assemble)
	grade(grade)
	rewrite(rewrite)
	generate(generate)
	citations(citations)
	__end__([<p>__end__</p>]):::last
	__start__ --> prepare;
	assemble --> grade;
	classify -.-> chitchat;
	classify -.-> retrieve;
	generate --> citations;
	grade -.-> generate;
	grade -.-> rewrite;
	prepare -. &nbsp;end&nbsp; .-> __end__;
	prepare -. &nbsp;continue&nbsp; .-> classify;
	retrieve --> assemble;
	rewrite --> retrieve;
	chitchat --> __end__;
	citations --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```
