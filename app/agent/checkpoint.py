"""
Checkpointer — LangGraph 的状态持久化

**作用**：把每一步的状态存下来，使工作流支持
  - 多轮会话（同一 `thread_id` 续接上下文）
  - 中断恢复（进程重启后继续）
  - 时间旅行调试（回放到任意一步）

**为什么用 SQLite**：
  - 依赖轻（stdlib sqlite3 + aiosqlite），零运维
  - 文件落盘，重启不丢
  - 与项目「单机单 worker」的定位一致（索引是进程内状态，本来就只能单 worker）

**降级策略**：SQLite 不可用时退回内存 checkpointer（进程内有效，重启丢）。
理由与项目的整体策略一致 —— 持久化失败不该让服务起不来。

⚠️ **并发约束**：SQLite 单文件在 Windows 下有写锁。本项目已限定单 worker
（见 docs/week10 的部署约束），所以不构成问题；但如果将来上多 worker，
这个 checkpointer 和进程内的 FAISS 索引会同时成为阻塞点。
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from app.config import settings

logger = logging.getLogger("rag_api.agent.checkpoint")


class CheckpointerHandle:
    """
    Checkpointer 的持有者。

    `AsyncSqliteSaver` 是异步上下文管理器，生命周期要覆盖整个应用运行期，
    所以用这个类持有底层连接，由 lifespan 负责 open/close。
    """

    def __init__(self, saver=None, *, kind: str = "none", path: Path | None = None):
        self.saver = saver
        self.kind = kind
        self.path = path

    @property
    def enabled(self) -> bool:
        return self.saver is not None

    def describe(self) -> str:
        if self.kind == "sqlite":
            return f"SQLite（{self.path}）"
        if self.kind == "memory":
            return "内存（重启后丢失）"
        return "未启用"


@asynccontextmanager
async def open_checkpointer(db_path: Path | None = None):
    """
    打开 checkpointer（异步上下文管理器）。

    用法：
        async with open_checkpointer() as cp:
            graph = build_self_rag_graph(bundle).compile(checkpointer=cp.saver)

    Yields:
        CheckpointerHandle —— `.saver` 可能为 None（完全不可用时）
    """
    path = Path(db_path) if db_path else settings.AGENT_CHECKPOINT_DB

    # ── 优先 SQLite ──
    try:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(path))
        saver = AsyncSqliteSaver(conn)
        await saver.setup()          # 建表（幂等）

        logger.info("Checkpointer: SQLite 已就绪（%s）", path)
        try:
            yield CheckpointerHandle(saver, kind="sqlite", path=path)
        finally:
            await conn.close()
        return

    except Exception as e:
        logger.warning("SQLite checkpointer 不可用，降级为内存: %s", e)

    # ── 降级：内存 ──
    try:
        from langgraph.checkpoint.memory import MemorySaver

        logger.warning("Checkpointer: 使用内存模式，进程重启后会话状态会丢失")
        yield CheckpointerHandle(MemorySaver(), kind="memory")
        return

    except Exception as e:
        logger.error("内存 checkpointer 也不可用，工作流将无状态运行: %s", e)

    # ── 完全不可用 ──
    yield CheckpointerHandle(None, kind="none")


def clear_checkpoints(path: Path | None = None) -> bool:
    """
    删除 checkpoint 数据库（调试用）。

    注意：这会丢掉所有会话历史，仅用于测试与排障。
    """
    p = Path(path) if path else settings.AGENT_CHECKPOINT_DB
    if p.exists():
        p.unlink()
        logger.info("已删除 checkpoint 数据库: %s", p)
        return True
    return False
