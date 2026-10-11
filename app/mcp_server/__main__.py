"""
MCP Server 入口 —— 以 stdio 传输方式运行

用法:
    python -m app.mcp_server
    python D:\\RAG项目\\app\\mcp_server\\__main__.py

────────────────────────────────────────────────────────────────
**为什么这里要在 import app.* 之前做那一堆路径处理**

MCP Server 会被 Claude Desktop / Inspector 以子进程方式启动，而它们的
**工作目录不可控**（Claude Desktop 的配置里 `cwd` 字段并非官方文档化的字段，
不能依赖）。这带来两个独立的问题：

1. **导入问题（致命）**：`python -m app.mcp_server` 从 `sys.path[0]`（= 启动时的
   CWD）解析 `app` 包。如果 CWD 不对，压根导入不了 —— 而且这发生在
   `__main__` 执行**之前**，所以 `os.chdir()` 在 `__main__` 里救不了它。
   解法只能是提前把项目根插进 `sys.path`，或者靠 `PYTHONPATH`。

2. **相对路径问题**：`settings.INDEX_DIR = Path("data/index")` 是相对路径，
   所以必须 `os.chdir(PROJECT_ROOT)` 才能命中索引。

**两者都要处理，缺一不可。**

────────────────────────────────────────────────────────────────
**stdio 纯净度**

stdio 模式下 **stdout 就是 JSON-RPC 通道**，任何写到 stdout 的内容都会
让客户端解析失败（症状是「服务器无响应」而不是明确报错，很难排查）。

防护三层：
  1. 日志显式指向 stderr 且 `force=True`（不依赖默认值，也不被 SDK 的
     basicConfig 覆盖成第二个 handler）
  2. `app/mcp_server/__init__.py` 保持 import-light（包级代码在 serving
     窗口之外执行，SDK 的 stdout 重定向保护不到）
  3. 重量级初始化全部放在 lifespan 里（serving 窗口内）

**不**做 `sys.stdout = sys.stderr` 这种全局重定向 —— SDK 已经接管了 wire，
抢它的 stdout 反而会破坏协议写入。（实测确认：faiss 的加载信息走 stderr，
app/ 目录下也没有裸 print，所以当前是干净的。）
"""
import logging
import os
import sys
from pathlib import Path

# ══════════════════════════════════════════════════════════════
# 路径处理 —— 必须在任何 app.* 导入之前
# ══════════════════════════════════════════════════════════════

# app/mcp_server/__main__.py → 上溯三级到项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 相对路径（data/index、uploads/ 等）依赖工作目录
try:
    os.chdir(PROJECT_ROOT)
except OSError as e:      # 极端情况：目录被删/无权限
    print(f"无法切换到项目根目录 {PROJECT_ROOT}: {e}", file=sys.stderr)

# ══════════════════════════════════════════════════════════════
# 日志配置 —— 显式 stderr + force，防止污染 JSON-RPC
# ══════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,     # ← 关键：绝不能是 stdout
    force=True,            # ← 覆盖已有配置，避免 SDK 再加一个 handler
)

logger = logging.getLogger("rag_api.mcp")

# ══════════════════════════════════════════════════════════════
# 启动
# ══════════════════════════════════════════════════════════════


def main() -> None:
    logger.info("=" * 56)
    logger.info("RAG MCP Server 启动")
    logger.info("  项目根: %s", PROJECT_ROOT)
    logger.info("  传输:   stdio")
    logger.info("=" * 56)

    from app.mcp_server.server import mcp

    # run() 默认 transport='stdio'
    mcp.run()


if __name__ == "__main__":
    main()
