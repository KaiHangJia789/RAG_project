# 第 12 周：LangGraph 条件分支与循环

2026.09.28-10.04 ｜ 目标：条件路由 + 循环重试 + Checkpointer + Node 级 Trace

## 一、条件分支：按查询类型路由

### 设计

`classify` 节点把问题分成三类，条件边据此走不同路径：

| 类型 | 路径 | 理由 |
|---|---|---|
| `chitchat` | **直答，跳过检索** | 闲聊不需要检索，省掉一次无意义的向量检索 + 一次 LLM 调用 |
| `factual` | 标准检索 | 事实通常集中在单个段落 |
| `reasoning` | 扩大检索范围 | 答案分散在多个段落，需要更大的 `final_k` |

分流靠**节点内改写 config** 实现，而不是建三条重复的子图 —— 后者会让流程图爆炸且难维护。

### 降级策略

分类调用失败或返回非法值时**默认 `factual`**（而非 chitchat）：

> 宁可多检索一次，也不要把真实问题误判成闲聊而不检索。

这是有意的不对称设计 —— 两类错误的代价不同：误判为 factual 只是浪费一次检索；误判为 chitchat 会让用户拿到一个「我是知识库助手」的无效回答。

### 实测

```
你好呀                          → chitchat   检索 0 条, 调用 2 次
什么是 Keyset 分页？              → factual    检索 3 条, 调用 2 次
Keyset 和 OFFSET 分页在高并发下哪个更好 → reasoning  检索 3 条, 调用 2 次
```

分类准确，闲聊路径确实跳过了检索。

见 [graph_branch.md](graph_branch.md)。

## 二、循环检索：检索 → 评分 → 重写 → 再检索

### 设计

```
retrieve → assemble → grade ─┬─(达标)→ generate → citations → END
   ↑                         └─(不达标)→ rewrite ──┘
   └───────────────────────────────────────┘
```

`grade` 节点用判官评估「检索到的上下文能否回答问题」，不达标则改写查询重新检索。

### 三重终止条件（缺一不可）

| 层级 | 机制 | 作用 |
|---|---|---|
| 业务上限 | `rewrite_count >= AGENT_MAX_REWRITES`（默认 2） | 正常退出路径 |
| 语义收敛 | 改写结果与上次相同 → 强制把 count 推到上限 | 防止「改写不动」的空转 |
| 平台兜底 | `recursion_limit=25`（每次 invoke 传入） | 即使业务逻辑写错也不会失控 |

**为什么需要第二重**：如果模型没能给出新的改写角度，会产生
`检索 → 评分不变 → 改写不变 → 再检索` 的死循环 —— 前两重都拦不住（计数在涨但查询没变），只有主动探测「查询是否实质变化」才能提前跳出。

判定用归一化比较（去空白与标点），因为模型可能只加了个空格或换了个标点。

### 阈值不复用 `RAG_MIN_SCORE`

`AGENT_RELEVANCE_THRESHOLD`（默认 0.5）是**判官的语义评分**，
而 `RAG_MIN_SCORE` 是 **FAISS 余弦分** —— 两者量纲完全不同。混用会导致
「以为在调检索阈值，实际在调判官阈值」的困惑。

### 实测

```
正常问题「什么是向量检索？」:
  相关性 0.70 ≥ 0.5 → 直接生成。1 条引用，tokens 3829+98

语料外问题「如何用 Rust 写高性能 HTTP 服务器？」:
  相关性 0.00 → 重写 #1 → 0.00 → 重写 #2 → 0.00 → 达上限 → 生成 → LLM 拒绝回答
  tokens 7733+99（重写循环的成本代价一目了然）
```

**注意 token 差异**：循环让成本翻倍。这正是 A/B 对比要量化的东西。

见 [graph_loop.md](graph_loop.md)。

## 三、Checkpointer：SQLite 状态持久化

`app/agent/checkpoint.py` 提供 `open_checkpointer()` 异步上下文管理器：

- 优先 SQLite（`data/agent/checkpoints.sqlite`），文件落盘、重启不丢
- 不可用时**降级为内存**（进程内有效），并打 warning
- 完全不可用时返回 `saver=None`，工作流无状态运行

降级策略与项目整体一致：**持久化失败不该让服务起不来**。

### 并发约束

SQLite 单文件在 Windows 下有写锁。本项目已限定单 worker（索引本就是进程内状态），
所以不构成问题。若将来上多 worker，SQLite checkpointer 和 FAISS 索引会**同时**成为阻塞点。

## 四、Node 级 LangFuse Trace

### 问题

LangGraph 的节点在各自的 asyncio 任务里执行（Pregel 的 superstep 调度）。
LangFuse 4.x 基于 OpenTelemetry contextvars，而 contextvar 在 `create_task` 时
是**复制而非共享** —— 兄弟节点能否自动归到同一 trace 取决于框架调度实现细节，**不可靠**。

### 解法：显式挂父，不依赖 contextvar 继承

```python
with tracer.root(input={"question": q}):        # 根 span，建立 trace
    trace_id, span_id = tracer.current_ids()     # 取出 id
    ...
# 每个节点的 span 显式挂到根 span 下：
with tracer.span("node:retrieve", as_type="retriever",
                 trace_context={"trace_id": trace_id, "parent_span_id": span_id}):
    ...
```

`as_type` 用语义化类型（`retriever` / `generation` / `evaluator` / `tool` / `chain`），
Dashboard 里一眼能看出哪个节点在干什么。

### 验证结果（这是计划里标记的最高未验证风险）

劫持 `start_as_current_observation` 记录实际产生的层级：

```
agent      agent:agent      trace=             parent=(继承)
chain      node:prepare     trace=209cd0b551eb parent=7039648d6601
retriever  node:retrieve    trace=209cd0b551eb parent=7039648d6601
chain      node:assemble    trace=209cd0b551eb parent=7039648d6601
generation node:generate    trace=209cd0b551eb parent=7039648d6601
chain      node:citations   trace=209cd0b551eb parent=7039648d6601

结论: 共 6 个 span, 归属 1 个 trace  → ✅ 全部嵌套在同一 trace 下
```

**显式 `trace_context` 方案确定性生效**，不依赖框架的 contextvar 传播行为。

### 无 LangFuse 时的降级

`AgentTracer` 在 `langfuse_enabled()` 为 False 时所有方法都是 no-op，
节点代码不需要任何判断。追踪失败也绝不影响主流程（span 创建已包 try）。

## 五、验收产物

| 要求 | 产出 |
|---|---|
| 条件分支 RAG Demo | `build_branching_graph()`，三种类型路由验证通过 |
| 循环检索 Demo | `build_self_rag_graph()`，三重终止条件验证通过 |
| LangGraph Trace 可视化截图 | `AgentTracer` 上报至 LangFuse，层级验证通过（见上） |
| Checkpointer | `open_checkpointer()`（SQLite + 内存降级） |

## 六、已知限制

1. **Trace 截图需手动获取**：数据已上报 LangFuse，截图需登录 Dashboard 取。
2. **相关性评分成本**：每次循环多一次判官调用 + 一次改写调用。`AGENT_MAX_REWRITES=2`
   意味着最坏情况每题多 4 次 LLM 调用。
3. **Checkpointer 未接入 API**：目前 `AgentRunner.run(thread_id=...)` 支持传会话 ID，
   但 REST 端点尚未暴露多轮会话接口（第 15 周前端重构时一并做）。
