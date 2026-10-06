# 第 11 周：LangGraph 入门 —— 基础 RAG → LangGraph 迁移笔记

2026.09.21-09.27 ｜ 目标：理解 State / Node / Edge / Conditional Edge，把线性 RAG 流水线改造成图

## 一、开工前的准入门槛：工具调用兼容性实测

第 13 周要给 Agent 加工具调用，而 DeepSeek 的思考模式与 function calling 是否兼容**没有把握**，所以先做了 spike（`scripts/spike_tool_calling.py`）。

**结论（实测）**：

| 验证项 | 结果 |
|---|---|
| `thinking=enabled` + `tools` 共存 | ✅ **可以共存** —— Agent 循环能保留思考模式 |
| `arguments` 字段类型 | ✅ 确认是 **JSON 字符串**（`'{"expression": "123 * 456"}'`），不是 dict |
| 多轮工具调用 + `reasoning_content` 回传 | ✅ 正常 |
| 不传 `reasoning_content` | ⚠️ 当前版本**也成功**（宽松），但仍按文档要求回传 |

**最关键的一条官方规则**（决定了架构选择）：

> 带 `tools` 参数时，历史 assistant 消息的 `reasoning_content` **必须原样回传**，
> 即使那一轮没有调用工具，否则 HTTP 400。

搜索证据显示 **langchain 的 `ChatDeepSeek` 恰恰丢掉了这个字段**（他们为此提了 PR #40254 修复）。
这正是本项目**不用 langchain-openai、自己扩展 LLMClient** 的原因 ——
已有的 `ChatMessage.to_openai_dict()` 已经正确处理了该字段，换栈等于重新引入这个坑。

## 二、迁移的核心思路：先抽步骤，再换编排

改造前 `RagPipeline.answer()` 是一个 90 行的单体方法，7 个步骤硬编码在一条直线里。无法复用，也无从插入分支或循环。

**做法是两步**：

1. **抽出原子步骤** → `app/rag/steps.py`
2. **两种编排并存** —— `RagPipeline`（顺序调用）与 `app/agent/`（图式调用）

```
app/rag/steps.py          ← 原子步骤（两边共用）
   ├─ is_index_ready(index) -> bool
   ├─ retrieve(index, query, config) -> list[IndexHit]
   ├─ assemble_context(hits) -> str
   ├─ generate_answer(llm, question, context, config) -> LLMResponse
   └─ attach_citations(answer_text, hits) -> CitationResult

app/rag/pipeline.py       ← 命令式编排：顺序调用上述步骤
app/agent/nodes.py        ← 图式编排：每个步骤包一个节点
```

**这样做的价值**：「普通 RAG vs LangGraph」的对比就是字面意义上的**同一批步骤、两种编排方式** —— 行为差异只来自编排逻辑本身，而不是实现细节。这让迁移笔记有了说服力。

重构后 **256 个既有测试全绿**，证明重构是纯内部改动。

## 三、代码结构差异对比

### 命令式（`RagPipeline`，改造前/后行为一致）

```python
async def answer(self, question, config=None, *, history=None) -> RagAnswer:
    base = RagAnswer(question=question, answer="", config=cfg)

    if not steps.is_index_ready(self.index_service):      # 步骤 1
        return _refuse(base, RefusalReason.INDEX_NOT_READY)

    hits = await steps.retrieve(self.index_service, question, cfg)   # 步骤 2
    if not hits:                                          # 步骤 3（分支写死在代码里）
        return _refuse(base, RefusalReason.BELOW_THRESHOLD)

    context = steps.assemble_context(hits)                # 步骤 4
    resp = await steps.generate_answer(self.llm, question, context, config=cfg)
    ...
```

**特征**：流程是**代码的执行顺序**。分支靠 `if`，无法观测中间状态，无法插入新环节而不改这个方法。

### 图式（`app/agent/graphs.py` + `nodes.py`）

```python
graph = StateGraph(AgentState)
graph.add_node("prepare",   _node("prepare",   make_prepare_node(bundle)))
graph.add_node("retrieve",  _node("retrieve",  make_retrieve_node(bundle)))
graph.add_node("assemble",  _node("assemble",  make_assemble_node(bundle)))
graph.add_node("generate",  _node("generate",  make_generate_node(bundle)))
graph.add_node("citations", _node("citations", make_citation_node(bundle)))

graph.add_edge(START, "prepare")
graph.add_conditional_edges("prepare", _route_after_prepare,
                            {"continue": "retrieve", "end": END})
graph.add_conditional_edges("retrieve", _route_after_retrieve,
                            {"continue": "assemble", "end": END})
graph.add_edge("assemble", "generate")
graph.add_edge("generate", "citations")
graph.add_edge("citations", END)
```

**特征**：流程是**声明式的图结构**。分支是显式的条件边（带名字的判定函数），状态在节点间传递且可观测，新增环节只需 `add_node` + `add_edge`。

### 差异总结

| 维度 | 命令式 | 图式 |
|---|---|---|
| 流程定义 | 代码执行顺序 | 显式图结构 |
| 分支 | `if/return` 硬编码 | `add_conditional_edges` + 判定函数 |
| 中间状态 | 局部变量，外部不可见 | State，可观测、可持久化 |
| 新增环节 | 改方法体 | `add_node` + `add_edge` |
| 流程图 | 需要手画，容易与实现脱节 | `draw_mermaid()` 从代码生成 |
| 断点续跑 | 不支持 | 配 checkpointer 即可（第 12 周） |

## 四、流程图

由 `python scripts/render_graphs.py` 从**实际代码**生成（`graph.get_graph().draw_mermaid()`），不是手绘 —— 改图结构后重跑脚本即可同步，永远不会与实现脱节。

见 [graph_linear.md](graph_linear.md)。结构：

```
__start__ → prepare ─┬─(continue)→ retrieve ─┬─(continue)→ assemble → generate → citations → __end__
                     └─(end)──────→ __end__  └─(end)──────→ __end__
```

虚线是条件边。两条短路路径分别是：
- **索引未就绪** → 直接结束（不检索、不调 LLM）
- **检索无命中** → 阈值层拒答（不调 LLM，这是成本优化的关键路径）

## 五、State 设计

`AgentState` 是 `TypedDict, total=False`，**一次性定义骨架**、按周使用：

```python
class AgentState(TypedDict, total=False):
    # 输入
    question: str
    original_question: str
    config: dict                       # RetrievalConfig.model_dump()
    # 第 11 周
    hits: list[dict]; context: str; answer: str; citations: list[dict]
    # 第 12 周
    query_type: str; relevance_score: float; rewrite_count: int
    rewrite_history: Annotated[list[str], operator.add]
    # 第 13 周
    messages: Annotated[list[dict], operator.add]
    tool_calls: list[dict]; tool_results: Annotated[list[dict], operator.add]
    # 诊断
    llm_call_count: int; prompt_tokens: int; completion_tokens: int
```

### 两个关键决策

**① State 里只放 dict，不放 Pydantic 模型。**
checkpointer 要序列化状态，Pydantic 模型在跨版本时易碎。约定「State 只存 dict，节点进出时用 `serialization.py` 转换」。

**② 消息列表用 `operator.add`，绝不用 `add_messages`。**

`add_messages` 是 LangChain 专用 reducer，会对 dict 做消息语义转换 —— **它会丢掉 `reasoning_content`**，直接触发上一节说的 DeepSeek HTTP 400。

```python
messages: Annotated[list[dict], operator.add]   # ✅ 纯拼接，字段原样保留
# messages: Annotated[list, add_messages]       # ❌ 会丢 reasoning_content
```

代价是失去消息 id 去重与 `RemoveMessage` —— 本项目不需要这两个能力。

## 六、验收产物

| 要求 | 产出 |
|---|---|
| LangGraph HelloWorld Demo | `AgentRunner` + `build_linear_graph()`，端到端验证通过 |
| 含可视化流程图 | `docs/week11/graph_linear.md`（代码生成，非手绘） |
| 基础 RAG→LangGraph 版迁移笔记 | 本文档 |
| 对比代码结构差异 | 见第三节 |

## 七、验证记录

```
线性图端到端（真实索引 359 条向量）:
  问题「什么是 Keyset 分页？」→ 正确作答，2 条引用，4162ms
  负样本「Kubernetes 调度」→ 拒答（below_threshold），且 LLM 调用次数 = 0 ✅

测试: 271 passed（256 既有 + 15 新增，重构零回归）
```

## 八、已知限制

1. **单 worker**：索引是进程内状态，`--workers N` 会让各进程索引不一致（Week10 已记录）。
2. **无流式输出**：`LLMClient` 不支持 stream，所以图的节点级输出无法逐 token 推送。
3. **线性图的"条件边"本质是短路**：真正的分支在第 12 周（按查询类型路由）。
