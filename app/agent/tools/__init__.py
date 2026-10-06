"""Agent 工具集"""
from app.agent.tools.base import Tool, ToolRegistry, ToolResult
from app.agent.tools.calculator import CalculatorTool
from app.agent.tools.db_query import DbQueryTool, UnsafeSqlError, validate_readonly_sql
from app.agent.tools.search import (
    LocalCorpusSearch,
    SearchBackend,
    SearchDocsTool,
    WebSearchTool,
)

__all__ = [
    "CalculatorTool",
    "DbQueryTool",
    "LocalCorpusSearch",
    "SearchBackend",
    "SearchDocsTool",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "UnsafeSqlError",
    "WebSearchTool",
    "build_default_registry",
    "validate_readonly_sql",
]


def build_default_registry(index_service=None, db=None, search_backend=None) -> ToolRegistry:
    """
    装配默认工具集。

    按依赖可用性**选择性注册** —— 索引未就绪就不注册检索工具（而不是
    注册一个永远失败的），这样模型不会把轮次浪费在注定失败的工具上。

    Args:
        index_service: 有则注册 search_documents 与 web_search
        db: 有则注册 query_metadata
        search_backend: web_search 的后端；None 时用 LocalCorpusSearch
    """
    tools: list[Tool] = [CalculatorTool()]

    if index_service is not None and getattr(index_service, "ready", False):
        tools.append(SearchDocsTool(index_service))
        tools.append(WebSearchTool(
            search_backend or LocalCorpusSearch(index_service)
        ))

    if db is not None:
        tools.append(DbQueryTool(db))

    return ToolRegistry(tools)
