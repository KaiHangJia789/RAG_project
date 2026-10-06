# 第 13 周：LangGraph Agentic RAG 完整实现

2026.10.05-10.11 ｜ 目标：Self-RAG 完整链路 + 工具调用 + A/B 评测 + 对抗性测试集

## 一、LLMClient 的工具调用扩展

### 缺口

开工前 `ChatMessage` 已建好 `tool_calls` / `tool_call_id` 字段且 `to_openai_dict()` 正确处理，
但整条链路缺最后一段：

- `_build_params()` 不传 `tools`
- `LLMResponse` 没有 `tool_calls` 字段
- `_parse_response()` **直接丢弃** `message.tool_calls`

### 改动（保持既有测试不破）

| 位置 | 改动 |
|---|---|
| `LLMResponse` | 加 `tool_calls: list[dict] \| None = None`（带默认值，向后兼容） |
| `_build_params()` | 加 `tools` / `tool_choice` / `parallel_tool_calls`，**仅显式传入才写入** |
| `generate_chat()` | **新增**：直接接收完整 messages 列表 |
| `generate()` | 改为构造 messages 后**委托**给 `generate_chat()` |
| `_parse_response()` | 提取 `message.tool_calls`（`model_dump()` 转纯 dict） |
| `ChatMessage.from_response()` | **新增工厂**：保证 assistant 消息带 `reasoning_content` |

### 为什么不给 `generate()` 加 tools 参数

`generate(system, user_message, history=...)` 的形态装不下 Agent 的消息流
（system + user + assistant-with-tool_calls + tool-result + ... 混杂），
所以新增 `generate_chat(messages, ...)`，`generate()` 保留原签名并委托给它。

**7 个既有调用点（pipeline / judge / experiments / tests）零改动。**

### 实测的 SDK 结构（决定解析代码怎么写）

```
ChatCompletionMessage.tool_calls: list[ChatCompletionMessageFunctionToolCall] | None
  └─ ChatCompletionMessageFunctionToolCall
       ├─ id: str                      # 回传 tool 消息时要带
       ├─ type: Literal["function"]
       └─ function: Function
            ├─ name: str
            └─ arguments: str          # ⚠️ JSON 字符串，不是 dict！
```

**`arguments` 是 JSON 字符串** —— 执行工具前必须 `json.loads`，且模型可能给出
非法 JSON 或空串（无参数工具）。解析容错统一在 `ToolRegistry._parse_arguments()` 处理。

## 二、工具集

| 工具 | 用途 | 关键约束 |
|---|---|---|
| `search_documents` | 知识库语义检索 | 索引未就绪时不注册（避免模型把轮次浪费在必然失败的工具上） |
| `calculator` | 数学计算 | **不用 `eval`**，AST 白名单求值 |
| `query_metadata` | 知识库统计 | **只读 + 表白名单**，不暴露结构化动作之外的 SQL |
| `web_search` | 外部搜索 | 可插拔后端，默认本地语料实现（无需 API key） |

### 安全边界（工具参数是模型生成的文本）

**计算器 —— 绝不用 `eval`**。AST 白名单只允许算术运算节点，
以下全部被拒绝（有测试锁死）：

```
__import__("os").system("whoami")     → 参数不是合法 JSON
open("/etc/passwd").read()            → 参数不是合法 JSON
eval("1+1") / exec("x=1")             → 未授权的函数
(lambda: 1)() / [x for x in range(3)] → 不支持的语法
```

**一个由测试暴露的真实 DoS 漏洞**：AST 白名单只防住了"执行任意代码"，
**没防住"消耗任意资源"**。`9**9**9` = 9^387420489，结果是约 3.7 亿位的整数，
CPython 会尝试完整算出它 —— 吃光内存并卡死进程（**无法中断**，实测把 pytest 挂死）。

修复是在**求值前**检查指数规模（上限 10000）与结果位数（上限 10 万位）：

```python
if isinstance(node.op, ast.Pow):
    self._guard_pow(left, right)    # 必须在 op(left, right) 之前
```

修复后所有爆炸性表达式在 **0ms** 内被拒绝，正常运算不受影响。

**数据库工具 —— 三条硬约束**：
1. 只允许 SELECT（正则拒绝 INSERT/UPDATE/DROP/TRUNCATE 等）
2. 表白名单（`documents` / `chunks`），**绝不允许触及 `users`**（含密码哈希）
3. 强制 LIMIT，防止全表扫描

`validate_readonly_sql()` 实现了完整校验并配单测 —— 当前走结构化动作不需要它，
但如果将来开放自定义 SQL，**必须先过这一关**。

### 可插拔的搜索后端

```python
class SearchBackend(Protocol):
    async def search(self, query: str, *, max_results: int = 5) -> list[dict]: ...

class LocalCorpusSearch:   # 默认：本地语料关键词匹配，零依赖
    ...
# 接真实 API（Tavily 等）时实现同一协议并注入，WebSearchTool 零改动
```

## 三、ReAct 与混合架构

### 图结构

```
agent ─┬─(有工具调用)→ tools → agent   ← ReAct 循环
       └─(无工具调用)→ finalize → citations → END
```

模型自主决定调哪个工具、调几次。轮次上限 `AGENT_TOOL_MAX_ROUNDS=4` 是硬约束。

### 一个真实发现：ReAct 模式的引用质量问题

实测中，即便在 system prompt 里**反复强调**「用 [n] 标注引用、禁用 Markdown」，
ReAct 模式下的模型仍会输出带 `##` 标题和 `**` 加粗的自由文本，**引用率为 0**。

而固定管线的 `rag_context_cited` prompt 能稳定产出结构化引用。

**解法是混合架构**：

> 让 Agent 循环负责**灵活地收集信息**（这是它的强项），
> 把**按格式产出带引用的答案**交回经过验证的标准管线（这是它的强项）。

`finalize` 节点在以下条件同时满足时才真正执行：
- 有检索命中（`hits` 非空）
- 原答案中没有任何 `[n]` 引用标记

代价是额外一次 LLM 调用，但换来可用的引用。

**修复前后对比**：

```
修复前: 引用 0: []
修复后: 引用 3: [(1, '01_rag_basics.md'), (2, '01_rag_basics.md'), (3, '01_rag_basics.md')]
```

### 引用来源的传递链

```
search_documents 工具
  → ToolResult.meta["raw_hits"]        （原始 IndexHit，不做二次回捞）
  → tool_executor 节点 _dedupe_hits()   （按 faiss_id 去重，按分数重排）
  → state["hits"]
  → citations 节点 attach_citations()   （[n] 编号映射回 Citation）
```

**为什么不在上层用关键词回捞**：初版实现用关键词二次匹配来重建 hits，
结果既不准又脆弱（引用为空）。改为让工具直接暴露原始命中后问题消失。

## 四、20 个对抗性/边界测试问题

独立文件 `data/eval/adversarial_eval_set.json`（**与 60 题常规集分开** ——
后者是 A/B 的规范基准，混入对抗题会破坏可比性）。

5 类 × 4 题：

| 类别 | 考察点 | 期望行为 |
|---|---|---|
| `premise` | 错误前提 / 反问 | 否定前提，而非顺着答 |
| `multihop` | 多跳（需拼接 ≥2 处信息） | 跨段/跨文档综合 |
| `out_of_scope` | 语料中确实无答案 | 拒答 |
| `ambiguous` | 指代不明 / 过度笼统 | 澄清或覆盖多解 |
| `injection` | 提示注入 | 拒绝泄露与服从 |

**自洽性校验**（`tests/test_adversarial_set.py`，17 项，零 API 成本）：

- 恰好 20 题、每类恰好 4 题、id 唯一
- **期望行为与类别自洽**：`refuse` 只用于 `out_of_scope`；
  `multihop` 必须标 ≥2 个来源（否则它根本不是多跳）；
  `premise` 必须标为可回答（答案就是「你的前提错了」）
- 引用的文档真实存在、标准答案足够详实、每题写明考察意图

这类校验拦截了「评测集本身的错误伪装成系统能力不足」—— 校验过程中确实
抓出两处数据问题（一个标准答案过于简略、一个多跳题只标了 1 个来源），已修正。

## 五、A/B 评测

### 让两条路径产出同一种结果

`app/evaluation/runners.py` 让 `basic` / `agent` / `react` 三种模式**都产出 `RagAnswer`**。

这是让 A/B 成立的关键 —— `RAGEvaluator.evaluate_one(result: RagAnswer)` **不需要适配层**，
两条路径的指标口径天然一致。若 Agent 产出自己的结果模型，适配层正是口径漂移的温床
（Week10 就踩过 `retrieved_docs` 语义不一致的坑）。

### 用法

```bash
python scripts/run_rag_eval.py --mode basic --week week13 --name ab_basic
python scripts/run_rag_eval.py --mode agent --week week13 --name ab_agent
python scripts/run_ab_eval.py --a ab_basic --b ab_agent
```

### 公平性说明

Agent 模式每题会多调用若干次 LLM（分类、评分、重写、工具轮），所以**延迟与 token 天然更高**。
直接比绝对值意义不大，有意义的是：

1. **各质量指标的差异** —— 尤其是零成本的确定性指标（Hit Rate / MRR）
2. **同等质量下的额外成本** —— 延迟倍率是多少？换来了什么？

对比结果见 [ab_report.md](ab_report.md)。

## 六、验收产物

| 要求 | 产出 |
|---|---|
| Agentic RAG 完整 Demo | `build_agentic_graph()` + `build_react_graph()`，端到端验证通过 |
| 基础 vs Agentic 指标对比表 | `docs/week13/ab_report.md` |
| 工具调用日志 | `state["tool_results"]` 累积记录 + 日志输出 |
| A/B 评测报告 | 同 ab_report.md（含逐题差异与失败案例） |
| 边界测试用例集 | `data/eval/adversarial_eval_set.json`（20 题）+ 校验测试 |

## 七、验证记录

```
工具安全: 54 项测试（含 9 项代码执行注入 + 6 项 SQL 写操作 + DoS 保护）
工具调用: calculator / query_metadata / search_documents 均端到端验证
混合架构: 引用从 0 条修复为 3 条正确映射
对抗集: 17 项自洽性校验通过
全量测试: 325 passed
```

## 八、已知限制

1. **`web_search` 默认是本地实现**：不具备真正的联网能力，只在语料范围内有效。
   接真实 API 需实现 `SearchBackend` 协议并注入。
2. **对抗性判据尚未自动化**：20 题的期望行为（如"是否否定前提"、"是否服从注入"）
   需要新的判官 prompt 才能自动评分，目前是人工判读 + 复用常规指标。
3. **ReAct 模式的引用依赖 finalize 补救**：模型本身不会主动产出规范引用，
   这多了一次 LLM 调用。
4. **工具轮次上限 4**：复杂多跳问题可能不够，但调大会显著增加成本与时延。
