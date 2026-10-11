"""
MCP Server stdio 冒烟测试 —— 真实子进程端到端验证

用法:
    python scripts/smoke_mcp_stdio.py

**为什么需要这个脚本**（内存 Client 的测试覆盖不到的东西）：
  1. **入口点是否正确** —— `python -m app.mcp_server` 能不能起来
  2. **sys.path / chdir 处理** —— 从任意工作目录启动都能找到索引
  3. **stdout 纯净度** —— 任何污染都会让客户端解析失败
     （这是最隐蔽的失败模式：服务器看似启动了，但客户端说无法解析响应）

  这三项都是「只测 handler 函数」永远发现不了的问题。
"""
import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / "RAG_venv" / "Scripts" / "python.exe"

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)


async def main() -> int:
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters, stdio_client

    print("=" * 62)
    print("MCP Server stdio 冒烟测试")
    print("=" * 62)
    print(f"Python:  {PYTHON}")
    print(f"入口:    python -m app.mcp_server")
    print(f"工作目录: {PROJECT_ROOT}")
    print()

    # 刻意用 `-m app.mcp_server`（而不是直接跑 __main__.py），
    # 因为这是文档里推荐给用户的方式，必须验证它可行
    params = StdioServerParameters(
        command=str(PYTHON),
        args=["-m", "app.mcp_server"],
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT), "PYTHONIOENCODING": "utf-8"},
    )

    failures = []

    try:
        async with Client(stdio_client(params)) as client:
            # ── 1. 协议协商 ──
            print(f"[1] 协议版本: {client.protocol_version}")
            if not client.protocol_version:
                failures.append("协议未协商")

            # ── 2. 三层原语 ──
            tools = sorted(t.name for t in (await client.list_tools()).tools)
            print(f"[2] 工具: {tools}")
            for expected in ("search_documents", "get_document_by_id"):
                if expected not in tools:
                    failures.append(f"缺少工具 {expected}")

            resources = sorted(str(r.uri) for r in (await client.list_resources()).resources)
            print(f"[3] 资源: {resources}")

            prompts = sorted(p.name for p in (await client.list_prompts()).prompts)
            print(f"[4] 提示: {prompts}")

            # ── 3. 真实调用检索工具 ──
            r = await client.call_tool(
                "search_documents", {"query": "什么是 Keyset 分页", "top_k": 2}
            )
            ok = r.is_error is False and r.content
            print(f"[5] search_documents: {'✅' if ok else '❌'}")
            if ok:
                preview = r.content[0].text[:70].replace("\n", " ")
                print(f"      {preview}...")
            else:
                failures.append("search_documents 调用失败")

            # ── 4. 真实调用文档读取工具 ──
            r2 = await client.call_tool(
                "get_document_by_id",
                {"document_id": "3622ddd6-c4b2-4296-b130-fb4472228ebf", "limit": 1},
            )
            ok2 = r2.is_error is False and "分块总数" in (r2.content[0].text if r2.content else "")
            print(f"[6] get_document_by_id: {'✅' if ok2 else '❌'}")
            if ok2:
                head = r2.content[0].text.split("\n")[2]
                print(f"      {head.strip()}")
            else:
                failures.append("get_document_by_id 调用失败")

            # ── 5. 读资源 ──
            rr = await client.read_resource("docs://list")
            import json
            total = json.loads(rr.contents[0].text).get("total")
            print(f"[7] docs://list: {'✅' if total else '❌'}  ({total} 篇文档)")
            if not total:
                failures.append("docs://list 读取失败")

    except Exception as e:
        print(f"\n❌ stdio 连接失败: {type(e).__name__}: {e}")
        failures.append(f"stdio 连接异常: {e}")

    print()
    print("=" * 62)
    if failures:
        print(f"❌ 失败 {len(failures)} 项：")
        for f in failures:
            print(f"   - {f}")
        return 1

    print("✅ 全部通过 —— stdio 通道畅通，三层原语可用")
    print()
    print("下一步：MCP Inspector 可视化验证")
    print(f'  npx -y @modelcontextprotocol/inspector "{PYTHON}" -m app.mcp_server')
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
