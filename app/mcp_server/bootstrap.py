"""
MCP Server 的服务装配

**为什么不复用 `app/dependencies/auth.py:init_services()`**：
  那个模块顶部 `import fastapi`，还会连带拉起 `app.rag.pipeline`、
  `DashscopeEmbeddingClient`、`ParsingService` 等整条 Web 依赖链。
  MCP Server 是个纯 stdio 进程，不需要（也不该）引入 Web 框架。

  所以这里照 `scripts/run_rag_eval.py` 的做法手工装配所需的最小集合。

**降级策略**（与项目一贯做法一致：外部依赖不可用不阻断启动）：
  - 索引加载失败 → 检索工具不可用，但服务仍能起来（`docs://list` 等仍可用）
  - 数据库连不上 → `get_document_by_id` 报错，但检索仍可用
  这样任何单一依赖故障都不会让整个 MCP Server 挂掉。
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.agent.tools import GetDocumentTool, SearchDocsTool, ToolRegistry
from app.config import settings

logger = logging.getLogger("rag_api.mcp.bootstrap")

# 项目根目录：app/mcp_server/bootstrap.py → 上溯两级
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class McpServices:
    """MCP Server 运行期持有的服务集合"""

    registry: ToolRegistry
    index_service: object | None = None
    db: object | None = None
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        """人类可读的状态摘要（启动时打到 stderr）"""
        parts = []
        if self.index_service is not None and getattr(self.index_service, "ready", False):
            parts.append(f"索引 {self.index_service.size} 条向量")
        else:
            parts.append("索引不可用")
        parts.append("数据库已连接" if self.db is not None else "数据库不可用")
        parts.append(f"工具 {len(self.registry)} 个")
        return " | ".join(parts)


# ═══════════════════════════════════════════════════════════════
# 静态 Resource 的服务访问通道
# ═══════════════════════════════════════════════════════════════
#
# **为什么需要这个模块级引用**：
#   MCP SDK 有一条硬约束 —— **静态资源（URI 里没有模板变量）不允许声明
#   `Context` 参数**，报错原文是：
#     "Resource 'docs://list' has no URI template variables, but the handler
#      declares a Context parameter. Context injection for static resources is
#      not supported."
#
#   于是静态资源拿不到请求上下文，也就取不到 lifespan 里装配的服务。
#   解法是让 lifespan 把服务写进这个模块级引用，静态资源从这里读。
#
#   动态资源（如 `docs://{id}/meta`）可以正常用 Context 注入，不需要这个。
#   两种写法各有适用场景，这里选择保持 `docs://list` 静态 ——
#   稳定 URI 对客户端更友好（可收藏、可硬编码）。

_current: McpServices | None = None


def set_current_services(services: McpServices | None) -> None:
    """由 lifespan 在启动时调用"""
    global _current
    _current = services


def get_current_services() -> McpServices | None:
    """静态 Resource 用这个取服务（动态资源请用 Context 注入）"""
    return _current


async def build_services(*, need_db: bool = True) -> McpServices:
    """
    装配 MCP Server 需要的服务。

    每一步都独立 try —— 任一依赖失败只记录警告，不抛异常。
    """
    warnings: list[str] = []
    index_service = None
    db = None
    tools = []

    # ── 1. 向量索引（检索工具的前提）──
    try:
        from app.embedding.dashscope_embedding import DashscopeEmbeddingClient
        from app.services.index_service import IndexService

        index_service = IndexService(DashscopeEmbeddingClient())
        index_service.load()
        if index_service.ready:
            tools.append(SearchDocsTool(index_service))
            logger.info("索引已加载：%d 条向量", index_service.size)
        else:
            msg = (
                f"索引为空（策略={settings.CHUNK_STRATEGY}），检索工具不可用。"
                f"请先运行：python scripts/build_index.py --strategy {settings.CHUNK_STRATEGY}"
            )
            warnings.append(msg)
            logger.warning(msg)
    except Exception as e:
        msg = f"索引加载失败，检索工具不可用：{e}"
        warnings.append(msg)
        logger.warning(msg)

    # ── 2. 数据库（文档读取工具的前提）──
    if need_db:
        try:
            from app.db.connection import Database

            db = Database()
            await db.connect(settings.DATABASE_DSN)
            tools.append(GetDocumentTool(db))
            logger.info(
                "数据库已连接：%s:%s/%s",
                settings.POSTGRES_HOST, settings.POSTGRES_PORT, settings.POSTGRES_DB,
            )
        except Exception as e:
            msg = f"数据库连接失败，文档读取工具不可用：{e}"
            warnings.append(msg)
            logger.warning(msg)
            db = None

    return McpServices(
        registry=ToolRegistry(tools),
        index_service=index_service,
        db=db,
        warnings=warnings,
    )


async def close_services(services: McpServices) -> None:
    """释放资源（连接池）"""
    if services.db is not None:
        try:
            await services.db.disconnect()
        except Exception as e:
            logger.warning("关闭数据库连接失败: %s", e)
