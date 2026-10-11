# 第 14 周：MCP Server

2026.10.12-10.18 ｜ 把 RAG 知识库通过 Model Context Protocol 暴露给外部 AI 客户端

## 一、MCP 协议架构：三层原语

MCP 把「AI 应用连接外部能力」抽象成三层原语，理解它们的区别是本周的核心：

| 原语 | 语义 | 谁发起 | 类比 | 本项目实现 |
|---|---|---|---|---|
| **Tool** | 模型主动调用的函数 | 模型 | 动词（做一件事） | `search_documents`、`get_document_by_id` |
| **Resource** | 客户端按需读取的数据 | 客户端 | 名词（读一份数据） | `docs://list`、`docs://stats` |
| **Prompt** | 预置的提示模板 | 用户选取 | 模板 | `rag_qa`、`summarize_document` |

一句话区分前两者：**Tool 是「做一件事」，Resource 是「读一份数据」**。
Tool 由模型决策何时调用；Resource 是客户端按 URI 寻址拉取，像读文件。

三者不是孤立的 —— 本项目的 `rag_qa` 提示词会引导模型先调 `search_documents`，
再用 `docs://list` 确认文档范围，体现「用协议表达工作流」。

## 二、架构主线：同一套工具，两种暴露方式

本周最有价值的架构点是**复用**。第 13 周为 LangGraph Agent 写的 `Tool` 抽象里：

```python
class SearchDocsTool(Tool):
    name = "search_documents"        # ← 已经与本周需求同名
    # parameters: query(required) + top_k(1-20)
```

与本周的 `search_documents(query, top_k)` 需求**完全一致**。所以本周没有重写任何检索逻辑：

```
                     ┌─ app/agent/  (LangGraph Agent，第 13 周)
app/agent/tools/ ────┤
  SearchDocsTool     └─ app/mcp_server/  (MCP Server，第 14 周)
  GetDocumentTool
```

**两个消费方共享同一份业务逻辑**（参数钳制、空值校验、索引未就绪提示、格式化）。
MCP 侧只加一层薄包装，把 `ToolResult` 转成 MCP 的返回/异常语义。

### 一个刻意的取舍：不复用 schema

MCP 的输入 schema 从**类型注解 + docstring** 生成，所以现有 `Tool.parameters`
property 用不上。但换来的是：**参数描述和约束直接写在函数签名上**，Claude 看到的
工具说明就是代码本身，不会出现「schema 改了实现没改」的漂移。

```python
@mcp.tool()
async def search_documents(
    query: Annotated[str, Field(description="语义完整的问句或关键词组合")],
    top_k: Annotated[int, Field(ge=1, le=20, description="返回条数，默认 5")] = 5,
    ctx: Context = None,
) -> str:
    """在已索引的知识库文档中做语义检索，返回最相关的文本片段（含来源标注）。"""
```

## 三、三个踩过的真实坑

### 坑 1：MCP SDK 2.x 是重写，不是改名

网上绝大多数教程是 1.x 的 FastMCP 写法，**照抄会直接 `ModuleNotFoundError`**：

| 1.x | 2.x |
|---|---|
| `from mcp.server.fastmcp import FastMCP` | `from mcp.server import MCPServer` |
| `FastMCP(name=..., port=...)` | `MCPServer(name=..., instructions=..., log_level=...)` |
| `ctx.fastmcp` | `ctx.mcp_server` |
| `get_context()` | 声明 `ctx: Context` 参数 |
| `FastMCPError` | `MCPServerError` |
| camelCase 属性（`inputSchema`） | **snake_case**（`input_schema`） |

传输方式也分开了：构造时**不传** `port`/`host`（传了报 `TypeError`），
改为 `mcp.run(transport="stdio")` —— 默认值就是 stdio。

**协议层的变化**：2026-07-28 修订**无握手、无会话**，每个请求自带协议版本；
服务端**不能**反向调用客户端（sampling / elicitation / roots 已废弃）。

### 坑 2：静态 Resource 不能声明 `Context` 参数

MCP SDK 有一条硬约束，报错原文很清楚：

```
ValueError: Resource 'docs://list' has no URI template variables, but the handler
declares a Context parameter. Context injection for static resources is not supported.
```

于是静态资源取不到请求上下文、拿不到 lifespan 装配的服务。
解法是让 lifespan 把服务写进 `bootstrap.get_current_services()` 这个模块级引用。

> **动态资源**（URI 带 `{param}`，如 `docs://{doc_id}/meta`）可以正常用 Context 注入，
> 那种情况下 `security=ResourceSecurity(...)` 才有意义 —— 它用于拦截客户端从
> URI 传入的路径穿越、绝对路径、空字节等恶意值。静态 URI 没有可提取的参数，
> 所以不需要它。

这里选择保持 `docs://list` 静态：稳定 URI 对客户端更友好（可收藏、可硬编码）。

### 坑 3：错误必须 `raise ToolError`，不能 `return` 错误字符串

这是 MCP 最容易踩、后果最隐蔽的坑：

```python
# ❌ 错的：is_error=False，模型以为调用成功了，把这句话当成答案
if not result.ok:
    return result.content

# ✅ 对的：is_error=True，模型知道失败了，可以重试或换个做法
if not result.ok:
    raise ToolError(result.content)
```

本项目的 `Tool.run()` 设计成**永不抛异常**（第 13 周为 Agent 循环做的，
让模型能从工具错误中自行纠正）。这个设计在 MCP 下**必须转换** ——
所以 `tools.py` 里的 `_unwrap()` 承担了这个映射：

```python
def _unwrap(result) -> str:
    if not result.ok:
        raise ToolError(result.content)
    return result.content
```

## 四、分页：不是优化，是必需

`get_document_by_id` 必须分页。实测数据：

```
student_hardbook.pdf: 325 个分块，中位长度 575 字符，总计约 19 万字符
```

一次返回全文会直接撑爆上下文窗口。所以：

- 默认 `limit=20`（约 11.5k 字符 ≈ 4-6k token）
- 硬上限 `limit=50`（约 29k 字符）
- 返回文本带**续取提示**，让模型知道还有内容且知道怎么拿

返回格式（`limit=2` 时）：

```
文档：student_hardbook.pdf
ID：3622ddd6-...  类型：.pdf  大小：1.32 MB  状态：ready
分块总数：325（策略 splitter）  当前：第 1–2 块（offset=0, limit=2，剩余 323 块）

[块 0 · 第 1 页]
前言
一、校训及其释义
...
（还有 323 块未显示。继续请调用 get_document_by_id(document_id="3622ddd6-...", offset=2, limit=2)）
```

读完后显示「（已显示全部内容）」，且头部不再显示「剩余 0 块」这类噪音。

### 另一个易漏点：必须按 chunk_strategy 过滤

005 迁移后的唯一约束是 `(document_id, chunk_strategy, chunk_index)` ——
同一文档可以有多套策略的分块共存。不过滤就会拿到**交错重复**的行。

实现是「先按当前策略取，为空则回退不过滤并如实报告实际策略」，
避免因为「DB 是用别的策略建的」而让工具完全不可用。

## 五、stdio 纯净度（最隐蔽的失败模式）

stdio 模式下 **stdout 就是 JSON-RPC 通道**。任何写到 stdout 的内容都会让客户端
解析失败 —— 而症状是「服务器无响应」而不是明确报错，排查方向极易跑偏。

防护三层：

1. **日志显式指向 stderr** 且 `force=True`（不被 SDK 的 `basicConfig` 覆盖成第二个 handler）
2. **`app/mcp_server/__init__.py` 保持 import-light**（只放 docstring）——
   包级代码在 serving 窗口之外执行，SDK 的 stdout 重定向保护不到
3. **重量级初始化全部放在 lifespan 里**（serving 窗口内）

**实测确认**：单独捕获 stdout 运行 `build_services()`，输出为空。
faiss 的加载信息（`Loading faiss with AVX2 support`）走的是 stderr ——
那是它在 C++ 层用 `fprintf(stderr, ...)` 输出的。`app/` 目录下也没有裸 `print()`
（`app/llm/client.py` 里那两处在 docstring 内）。

> 刻意**不**做 `sys.stdout = sys.stderr` 这种全局重定向 —— SDK 已经接管了 wire，
> 抢它的 stdout 反而会破坏协议写入。

## 六、工作目录：两个独立的问题

MCP Server 被 Claude Desktop / Inspector 以子进程启动，**工作目录不可控**
（Claude Desktop 的 `cwd` 字段并非官方文档化的字段，不能依赖）。这带来两个独立问题：

| 问题 | 后果 | 解法 |
|---|---|---|
| **导入** | `-m app.mcp_server` 从 `sys.path[0]`（=启动时 CWD）解析 `app` 包，CWD 不对就导入失败 | 在 `__main__.py` **顶部**把项目根插进 `sys.path` |
| **相对路径** | `INDEX_DIR = Path("data/index")` 是相对路径，找不到索引 | `os.chdir(PROJECT_ROOT)` |

**关键点：`os.chdir` 救不了导入问题** —— 导入发生在 `__main__` 执行**之前**。
所以两者都要处理，顺序不能反。

## 七、Claude Desktop 配置

### 配置片段（请自行粘贴）

配置文件有两个候选路径，**两个都试**（探查发现你机器上两处都有文件）：

| 路径 | 修改时间 |
|---|---|
| `%APPDATA%\Claude\claude_desktop_config.json` | 2026-08-22 |
| `%LOCALAPPDATA%\Claude\claude_desktop_config.json` | 2026-10-07（含 `deploymentMode: "3p"`） |

在文件里加上 `mcpServers` 段（**与现有内容合并，不要整个替换**）：

```json
{
  "mcpServers": {
    "rag-knowledge-base": {
      "command": "D:\\RAG项目\\RAG_venv\\Scripts\\python.exe",
      "args": ["-m", "app.mcp_server"],
      "env": {
        "PYTHONPATH": "D:\\RAG项目",
        "PYTHONIOENCODING": "utf-8"
      }
    }
  }
}
```

> `cwd` 字段不是官方文档化字段，**不要依赖**。上面的 `-m app.mcp_server` 形式
> 配合 `__main__.py` 里的自定位逻辑，已经能在任意工作目录下工作。

### 操作步骤

1. 备份原配置文件（改坏了能还原）
2. 粘贴上面的 JSON 片段（合并进现有内容）
3. **完全退出** Claude Desktop（不是关窗口，要退出进程）
4. 重新启动 —— 配置只在启动时读取
5. 确认工具图标出现，服务名 `rag-knowledge-base`
6. 试这两句：
   - 「用 search_documents 查一下知识库里关于 Keyset 分页的内容」
   - 「用 get_document_by_id 读取 student_hardbook.pdf 的前 20 个分块」

**验证要点**：回答里应包含**只存在于该文档中的内容**（如校训「厚德、博学、求真、至善」），
而不是模型的既有知识 —— 这才能证明工具真的被调用了。

### 如果没生效

1. 看 Claude Desktop 的 MCP 日志（Settings → Developer）
2. 用同一条命令跑 MCP Inspector，隔离「配置问题」与「代码问题」：

```bash
npx -y @modelcontextprotocol/inspector "D:\RAG项目\RAG_venv\Scripts\python.exe" -m app.mcp_server
```

## 八、验证记录

```
stdio 端到端（真实子进程，scripts/smoke_mcp_stdio.py）:
  [1] 协议版本: 2026-07-28
  [2] 工具: ['get_document_by_id', 'search_documents']        ✅
  [3] 资源: ['docs://list', 'docs://stats']                   ✅
  [4] 提示: ['rag_qa', 'summarize_document']                  ✅
  [5] search_documents 实际调用                                ✅
  [6] get_document_by_id 分页返回（325 块，正确显示第 1-1 块）   ✅
  [7] docs://list 读取（8 篇文档）                             ✅

错误语义（4 项全部 is_error=True，模型能感知失败）:
  空 query / 不存在的 ID / 非法 UUID 格式 / 超限参数            ✅

stdout 纯净度: 单独捕获 stdout 为空，无污染                      ✅

测试: 368 passed（342 既有 + 26 新增）
依赖: pip check 无冲突
```

## 九、验收产物对照

| 要求 | 产出 | 状态 |
|---|---|---|
| 理解 MCP 协议架构（Server/Client/Resource/Tool/Prompt） | 本文档第一节 + 三层原语全部实现 | ✅ |
| 搭建 MCP Server 基础框架（Python SDK） | `app/mcp_server/`（SDK 2.3.0） | ✅ |
| 实现 Tool：`search_documents(query, top_k)` | 复用第 13 周实现，加 MCP 包装 | ✅ |
| 实现 Tool：`get_document_by_id(id)` | 新增 `GetDocumentTool`（含分页） | ✅ |
| 用 Claude Desktop / MCP Inspector 测试 | Inspector 命令已给出；**Desktop 需你配置后验证** | ⚠️ |
| **MCP Server 可被 Claude Desktop 调用** | 配置片段 + 步骤已给出 | ⚠️ 待你操作 |
| 两个 Tool 正常工作 | stdio 端到端实测通过 | ✅ |

> **关于 Claude Desktop 的两点提醒**：
> 1. 你的 Desktop 是 `deploymentMode: "3p"`（第三方部署模式），
>    该模式**可能**限制自定义 MCP server。若被限制，验收以 MCP Inspector 为准。
> 2. 当前运行的 `claude` 进程实际是 **VS Code 扩展的原生二进制**，
>    不是 Claude Desktop 本体 —— 两者是不同的应用。

## 十、已知限制

1. **`docs://list` 依赖向量索引**：索引未构建时该资源返回错误。这是刻意的 ——
   它从索引 sidecar 读而非查数据库，好处是 Postgres 挂掉时依然可用。
2. **`get_document_by_id` 依赖数据库**：DB 不可用时该工具报错（检索工具不受影响）。
3. **无认证**：stdio 是本地进程间通信，不存在网络暴露面，所以未做鉴权。
   若要改成 HTTP 传输（`streamable-http`）必须补认证 —— MCP 2.x 的
   `MCPServer` 支持 `token_verifier` / `auth_server_provider`，但本周未涉及。
4. **分页上限 50 块**：极少数需要一次读完整份大文档的场景会需要多次调用。
   这是保护上下文窗口的必要代价。
