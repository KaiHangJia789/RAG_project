"""
MCP Tool 注册 —— 薄包装现有 Tool 抽象

**复用策略：复用业务逻辑，不复用 schema。**

  MCP 的输入 schema 从**类型注解 + docstring** 生成，所以现有 `Tool.parameters`
  property 用不上。但 `_run()` 里的业务逻辑（参数钳制、空值校验、
  索引未就绪提示、格式化）全部复用 —— 通过 `ToolRegistry.execute()` 调用，
  两个消费方（LangGraph Agent / MCP Server）共享同一份实现。

**最关键的一行是 `raise ToolError(...)`。**

  MCP 的错误语义有个容易踩的坑：handler **返回**错误字符串时，
  结果里 `is_error=False`，模型会以为工具调用成功了、把那句话当成答案。
  只有 **raise ToolError** 才会让 `is_error=True`，模型才知道该重试或换做法。

  所以下面的包装统一做：
      if not result.ok: raise ToolError(result.content)
"""
import logging
from typing import Annotated

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from app.mcp_server.bootstrap import McpServices

logger = logging.getLogger("rag_api.mcp.tools")


def _services(ctx: Context) -> McpServices:
    """
    从请求上下文取服务。

    服务在 lifespan 里装配，通过 `request_context.lifespan_context` 取用 ——
    这是 MCP 2.x 的正规做法（1.x 的 `get_context()` 已移除）。
    """
    return ctx.request_context.lifespan_context


def _unwrap(result) -> str:
    """
    把 `ToolResult` 转成 MCP 的返回值，失败时抛 `ToolError`。

    这是本模块最重要的一行逻辑 —— 见文件头的说明。
    """
    if not result.ok:
        raise ToolError(result.content)
    return result.content


def register_tools(mcp) -> None:
    """注册全部 MCP Tool"""

    @mcp.tool()
    async def search_documents(
        query: Annotated[
            str,
            Field(description="语义完整的问句或关键词组合，例如「什么是 Keyset 分页」"),
        ],
        top_k: Annotated[
            int,
            Field(ge=1, le=20, description="返回条数，默认 5，上限 20"),
        ] = 5,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """
        在已索引的知识库文档中做语义检索，返回最相关的文本片段（含来源标注）。

        当你需要查找事实、定义、步骤等具体信息时使用。
        可以用不同的说法多次调用以获取更全面的信息。
        """
        services = _services(ctx)
        # 走既有 Tool 实现：自动继承参数钳制、空查询校验、索引未就绪提示
        result = await services.registry.execute(
            "search_documents", {"query": query, "top_k": top_k}
        )
        return _unwrap(result)

    @mcp.tool()
    async def get_document_by_id(
        document_id: Annotated[
            str,
            Field(description="文档 UUID（可从 docs://list 资源获取）"),
        ],
        offset: Annotated[
            int,
            Field(ge=0, description="起始分块序号，从 0 开始，默认 0"),
        ] = 0,
        limit: Annotated[
            int,
            Field(ge=1, le=50, description="返回分块数，默认 20，上限 50"),
        ] = 20,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """
        按文档 ID 读取文档的元信息与分块正文，支持分页。

        用于回答「这份文档讲了什么」「文档有多少内容」这类需要通读的问题。
        大文档必须分页读取 —— 返回结果末尾会提示如何取下一页。
        """
        services = _services(ctx)
        result = await services.registry.execute(
            "get_document_by_id",
            {"document_id": document_id, "offset": offset, "limit": limit},
        )
        return _unwrap(result)

    logger.debug("已注册 MCP 工具: search_documents, get_document_by_id")
