"""
MCP Resource 注册

**Resource 与 Tool 的区别**（这是理解 MCP 协议的关键）：
  - **Tool** 是模型**主动调用**的函数，返回内容给模型看、可能有副作用
  - **Resource** 是客户端**按需读取**的数据，按 URI 寻址，像读文件

  一句话：Tool 是「做一件事」，Resource 是「读一份数据」。

**这里有个 SDK 的硬约束**（踩过一次，记录在此）：

  静态资源（URI 里没有模板变量，如 `docs://list`）**不能声明 `Context` 参数**：
  ```
  ValueError: Resource 'docs://list' has no URI template variables, but the
  handler declares a Context parameter. Context injection for static resources
  is not supported.
  ```
  所以静态资源取不到请求上下文，拿不到 lifespan 里装配的服务 ——
  只能通过 `bootstrap.get_current_services()` 这个模块级引用读。

  **动态资源**（URI 带 `{param}`，如 `docs://{doc_id}/meta`）则可以正常用
  `ctx: Context` 注入。两种写法各有场景，这里选静态是因为稳定 URI 对客户端
  更友好（可收藏、可硬编码）。
"""
import json
import logging

from mcp.server.mcpserver import Context

from app.mcp_server.bootstrap import McpServices, get_current_services

logger = logging.getLogger("rag_api.mcp.resources")


def register_resources(mcp) -> None:
    """注册 MCP Resource"""

    # ── 静态资源：知识库文档清单 ──
    @mcp.resource(
        "docs://list",
        name="知识库文档清单",
        description=(
            "列出知识库中的所有文档及其分块数量。"
            "用于发现有哪些可用文档，以及获取文档 ID 以便读取正文。"
        ),
        mime_type="application/json",
    )
    def list_documents() -> str:
        """
        列出知识库中的所有文档。

        数据来源是向量索引的元数据（sidecar），**不查数据库** ——
        所以即使 Postgres 不可用，这个资源依然能读。这是刻意的设计：
        让「发现有哪些文档」这个最基础的能力不被任何单一依赖卡住。
        """
        services = get_current_services()
        if services is None:
            return _err("服务尚未初始化")

        index = services.index_service
        if index is None or not getattr(index, "ready", False):
            return _err(
                "索引未就绪，无法列出文档。"
                "请先在项目根运行：python scripts/build_index.py"
            )

        # 从索引记录聚合出文档清单（按 document_id 去重）
        docs: dict[str, dict] = {}
        for rec in index.all_records():
            entry = docs.setdefault(rec.document_id, {
                "document_id": rec.document_id,
                "filename": rec.filename,
                "chunks": 0,
                "pages": set(),
            })
            entry["chunks"] += 1
            if rec.page_number:
                entry["pages"].add(rec.page_number)

        payload = {
            "total": len(docs),
            "strategy": index.strategy,
            "documents": [
                {
                    "document_id": d["document_id"],
                    "filename": d["filename"],
                    "chunks": d["chunks"],
                    "pages": len(d["pages"]) or None,
                    "hint": (
                        f'用 get_document_by_id(document_id="{d["document_id"]}") 读正文'
                    ),
                }
                for d in sorted(docs.values(), key=lambda x: -x["chunks"])
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    # ── 静态资源：知识库统计 ──
    @mcp.resource(
        "docs://stats",
        name="知识库统计",
        description="知识库规模统计：文档数、分块数、索引策略、可用工具。",
        mime_type="application/json",
    )
    def kb_stats() -> str:
        """知识库规模统计（同样来自索引元数据，不依赖数据库）"""
        services = get_current_services()
        if services is None:
            return _err("服务尚未初始化")

        index = services.index_service
        if index is None:
            return _err("索引服务不可用")

        payload = {
            **index.stats(),
            "db_available": services.db is not None,
            "tools": services.registry.names(),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    logger.debug("已注册 MCP 资源: docs://list, docs://stats")


def _err(msg: str) -> str:
    """统一的错误载荷（Resource 不能 raise ToolError，返回 JSON 即可）"""
    return json.dumps({"error": msg}, ensure_ascii=False, indent=2)
