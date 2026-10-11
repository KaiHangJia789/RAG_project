"""
MCP Server 定义

**三层原语**（MCP 协议的核心概念，本周任务要求理解）：
  - **Tool**     模型主动调用的函数 —— `search_documents` / `get_document_by_id`
  - **Resource** 客户端读取的数据（被动、按 URI 寻址）—— `docs://list`
  - **Prompt**   预置的提示模板（用户可选取）—— `rag_qa`

**生命周期**：
  重量级初始化（连数据库、加载 FAISS 索引）放在 `lifespan` 里，**不放模块级**。
  原因：stdio 模式下 SDK 只在 serving 窗口内保护 stdout，
  模块级代码在窗口之外执行，任何输出都会污染 JSON-RPC 通道。

**工具注册方式**：
  刻意用**独立的 `ToolRegistry`**，而不是复用 `build_default_registry()` ——
  后者会注册 calculator / web_search / query_metadata 等非本周需求的工具，
  且改动它会影响第 13 周 Agent 的既有行为。
"""
import logging
from contextlib import asynccontextmanager

from mcp.server import MCPServer

from app.mcp_server.bootstrap import (
    McpServices,
    build_services,
    close_services,
    set_current_services,
)

logger = logging.getLogger("rag_api.mcp.server")

SERVER_NAME = "rag-knowledge-base"

# 给模型的全局指令 —— 这是 MCP 提供的「引导模型怎么用这些工具」的正规位置
INSTRUCTIONS = """这是 RAG 项目（D:\\RAG项目）的知识库服务，包含技术文档与一份学生手册。

工具使用建议：
- 查找具体事实、定义、步骤 → 先用 search_documents 做语义检索
- 需要通读某份文档、或检索到的片段不够完整 → 用 get_document_by_id 读原文
  （大文档必须分页读取，返回结果会提示如何取下一页）
- 不要在没检索的情况下凭已有知识回答知识库相关问题
"""


def create_server(*, lifespan=None) -> MCPServer:
    """
    构造 MCPServer 实例。

    Args:
        lifespan: 自定义生命周期（测试时注入假服务用）。
            None 时用真实的 `app_lifespan`。
    """
    mcp = MCPServer(
        name=SERVER_NAME,
        instructions=INSTRUCTIONS,
        version="0.1.0",
        # WARNING 级别：避免 SDK 自身的 INFO 日志干扰（日志走 stderr，安全但吵）
        log_level="WARNING",
        lifespan=lifespan or app_lifespan,
    )

    # 三层原语各自注册（分开成模块，便于单独阅读与测试）
    from app.mcp_server.prompts import register_prompts
    from app.mcp_server.resources import register_resources
    from app.mcp_server.tools import register_tools

    register_tools(mcp)
    register_resources(mcp)
    register_prompts(mcp)

    return mcp


@asynccontextmanager
async def app_lifespan(server: MCPServer):
    """
    真实生命周期：启动时装配服务，关闭时释放。

    **服务有两种取用方式**（因为 SDK 的限制，两者都需要）：
      - 动态处理器（Tool、带模板变量的 Resource、Prompt）：
        通过 `ctx.request_context.lifespan_context` 注入
      - **静态 Resource**：不能声明 Context 参数（SDK 硬约束），
        只能从 `bootstrap.get_current_services()` 这个模块级引用读

    装配失败不会抛异常（`build_services` 内部逐项 try），
    所以服务总能起来，只是部分工具可能不可用。
    """
    logger.info("MCP Server 启动中，装配服务 ...")
    services = await build_services()
    set_current_services(services)      # ← 供静态 Resource 读取
    logger.info("服务就绪：%s", services.describe())
    for w in services.warnings:
        logger.warning("  ⚠️ %s", w)

    try:
        yield services
    finally:
        logger.info("MCP Server 关闭中 ...")
        set_current_services(None)
        await close_services(services)


# 模块级实例 —— CLI（`mcp dev app/mcp_server/server.py`）需要能找到它
mcp = create_server()
