"""
MCP Server 测试

分两层：
  1. `GetDocumentTool` 的业务逻辑（用假 Repository，不碰数据库）
  2. MCP Server 的整体行为（用 SDK 的内存 Client，不起子进程）

**注意不要用 `conftest.FakeConnection` 测 chunk 查询** ——
它的 `fetch()` 只识别 `user_id = $N` 和 `LIMIT $N`，会**静默忽略**
`WHERE document_id = $1` 和 `ORDER BY chunk_index`，返回表里所有行。
那样测试会"通过"但实际什么都没验证。所以下面用假 Repository。
"""
import json

import pytest

from app.agent.tools import GetDocumentTool, ToolRegistry


# ═══════════════════════════════════════════════════════════════
# 假 Repository（鸭子类型）
# ═══════════════════════════════════════════════════════════════

class FakeDocumentRepo:
    def __init__(self, doc: dict | None = None):
        self._doc = doc
        self.queries: list[str] = []

    async def get_by_id(self, conn, doc_id):
        self.queries.append(doc_id)
        if self._doc is None or self._doc.get("id") != doc_id:
            return None
        return self._doc


class FakeChunkRepo:
    """
    模拟 `ChunkRepository.fetch_by_document(conn, doc_id, strategy)`。

    记录每次调用的 strategy，用于验证「先按策略过滤」的逻辑。
    """

    def __init__(self, rows_by_strategy: dict[str | None, list[dict]]):
        self._rows = rows_by_strategy
        self.calls: list[tuple[str, str | None]] = []

    async def fetch_by_document(self, conn, document_id, strategy=None, **kw):
        self.calls.append((document_id, strategy))
        return self._rows.get(strategy, [])


class FakePool:
    def acquire(self):
        class _Ctx:
            async def __aenter__(self_inner):
                return object()
            async def __aexit__(self_inner, *a):
                pass
        return _Ctx()


class FakeDB:
    def __init__(self):
        self.pool = FakePool()


def make_doc(doc_id="doc-1", filename="测试文档.md", **kw) -> dict:
    return {
        "id": doc_id,
        "filename": filename,
        "file_type": ".md",
        "file_size": kw.get("file_size", 2048),
        "status": kw.get("status", "ready"),
    }


def make_chunks(n: int, strategy: str = "splitter", start: int = 0) -> list[dict]:
    return [
        {
            "id": f"chunk-{i}", "document_id": "doc-1",
            "chunk_index": i, "chunk_strategy": strategy,
            "chunk_text": f"第 {i} 块的内容", "page_number": i // 3 + 1,
        }
        for i in range(start, start + n)
    ]


def make_tool(doc=None, rows=None, **kw) -> GetDocumentTool:
    return GetDocumentTool(
        FakeDB(),
        doc_repo=FakeDocumentRepo(doc),
        chunk_repo=FakeChunkRepo(rows or {}),
        **kw,
    )


# ═══════════════════════════════════════════════════════════════
# GetDocumentTool 业务逻辑
# ═══════════════════════════════════════════════════════════════

class TestGetDocumentBasics:
    @pytest.mark.asyncio
    async def test_returns_metadata_and_chunks(self):
        tool = make_tool(doc=make_doc(), rows={"splitter": make_chunks(5)})
        r = await tool.run(document_id="doc-1")

        assert r.ok
        assert "测试文档.md" in r.content
        assert "分块总数：5" in r.content
        assert "第 0 块的内容" in r.content
        assert r.meta["total_chunks"] == 5

    @pytest.mark.asyncio
    async def test_missing_document_id_rejected(self):
        tool = make_tool()
        r = await tool.run(document_id="   ")
        assert not r.ok and r.error == "missing_id"

    @pytest.mark.asyncio
    async def test_document_not_found(self):
        tool = make_tool(doc=make_doc("other-id"), rows={"splitter": []})
        r = await tool.run(document_id="doc-1")
        assert not r.ok and r.error == "document_not_found"
        assert "未找到" in r.content

    @pytest.mark.asyncio
    async def test_db_unavailable(self):
        tool = GetDocumentTool(None)
        r = await tool.run(document_id="doc-1")
        assert not r.ok and r.error == "db_unavailable"


class TestPagination:
    """分页是必需的：大文档（如 325 块）全文会撑爆上下文"""

    @pytest.mark.asyncio
    async def test_first_page_hints_continuation(self):
        tool = make_tool(doc=make_doc(), rows={"splitter": make_chunks(50)})
        r = await tool.run(document_id="doc-1", offset=0, limit=10)

        assert r.ok
        assert "第 1–10 块" in r.content
        assert "剩余 40 块" in r.content
        # 必须告诉模型怎么续取，否则它会以为已经读完了
        assert "offset=10" in r.content

    @pytest.mark.asyncio
    async def test_middle_page(self):
        tool = make_tool(doc=make_doc(), rows={"splitter": make_chunks(50)})
        r = await tool.run(document_id="doc-1", offset=20, limit=5)

        assert "第 21–25 块" in r.content
        assert "offset=20" in r.content
        assert "第 20 块的内容" in r.content

    @pytest.mark.asyncio
    async def test_last_page_says_all_shown(self):
        tool = make_tool(doc=make_doc(), rows={"splitter": make_chunks(12)})
        r = await tool.run(document_id="doc-1", offset=10, limit=10)

        assert "已显示全部内容" in r.content
        assert "剩余" not in r.content

    @pytest.mark.asyncio
    async def test_small_document_fits_in_one_page(self):
        """10 块的小文档应一次返回完整内容"""
        tool = make_tool(doc=make_doc(), rows={"splitter": make_chunks(10)})
        r = await tool.run(document_id="doc-1")     # 默认 limit=20

        assert r.ok
        assert "已显示全部内容" in r.content
        assert "剩余" not in r.content

    @pytest.mark.asyncio
    async def test_offset_beyond_end(self):
        tool = make_tool(doc=make_doc(), rows={"splitter": make_chunks(5)})
        r = await tool.run(document_id="doc-1", offset=100, limit=10)
        # 超出范围时给空页 + 提示，而不是报错 —— 模型能自己修正
        assert r.ok
        assert "没有分块" in r.content or "已显示全部" in r.content

    @pytest.mark.asyncio
    async def test_limit_clamped_to_max(self):
        """模型可能给超大 limit，工具层必须兜底（MCP schema 也会拦，这里是第二道）"""
        tool = make_tool(doc=make_doc(), rows={"splitter": make_chunks(100)}, max_limit=50)
        r = await tool.run(document_id="doc-1", limit=9999)
        assert r.meta["limit"] == 50

    @pytest.mark.asyncio
    async def test_invalid_pagination_values_fall_back(self):
        tool = make_tool(doc=make_doc(), rows={"splitter": make_chunks(10)})
        r = await tool.run(document_id="doc-1", offset="abc", limit="xyz")
        assert r.ok
        assert r.meta["offset"] == 0
        assert r.meta["limit"] == tool.default_limit

    @pytest.mark.asyncio
    async def test_negative_offset_clamped(self):
        tool = make_tool(doc=make_doc(), rows={"splitter": make_chunks(10)})
        r = await tool.run(document_id="doc-1", offset=-5)
        assert r.meta["offset"] == 0


class TestStrategyFiltering:
    """
    唯一约束是 (document_id, chunk_strategy, chunk_index)，
    同一文档可以有多套策略的分块共存。不过滤会拿到交错重复的行。
    """

    @pytest.mark.asyncio
    async def test_filters_by_configured_strategy(self):
        chunk_repo = FakeChunkRepo({"splitter": make_chunks(3, "splitter")})
        tool = GetDocumentTool(
            FakeDB(), doc_repo=FakeDocumentRepo(make_doc()),
            chunk_repo=chunk_repo, strategy="splitter",
        )
        await tool.run(document_id="doc-1")

        # 第一次调用应该带 strategy
        assert chunk_repo.calls[0] == ("doc-1", "splitter")

    @pytest.mark.asyncio
    async def test_falls_back_when_strategy_empty(self):
        """
        当前策略下无数据时回退为不过滤，并如实报告用了哪些策略 ——
        避免因为「DB 是用别的策略建的」而让工具完全不可用。
        """
        rows_none = make_chunks(4, "paragraph")
        chunk_repo = FakeChunkRepo({None: rows_none})     # 只有"不过滤"能拿到
        tool = GetDocumentTool(
            FakeDB(), doc_repo=FakeDocumentRepo(make_doc()),
            chunk_repo=chunk_repo, strategy="splitter",
        )
        r = await tool.run(document_id="doc-1")

        assert r.ok
        assert "paragraph" in r.content          # 如实报告实际策略
        assert r.meta["strategy"] == "paragraph"
        assert ( "doc-1", None) in chunk_repo.calls      # 发生了回退

    @pytest.mark.asyncio
    async def test_multi_strategy_rows_sorted_stably(self):
        """多策略共存时按 (strategy, chunk_index) 排序，保证输出稳定"""
        rows = make_chunks(2, "sentence") + make_chunks(2, "paragraph")
        tool = GetDocumentTool(
            FakeDB(), doc_repo=FakeDocumentRepo(make_doc()),
            chunk_repo=FakeChunkRepo({None: rows}), strategy="splitter",
        )
        r = await tool.run(document_id="doc-1", limit=10)
        assert r.ok
        # 用了组合策略名
        assert "+" in r.meta["strategy"]


class TestFormatting:
    @pytest.mark.asyncio
    async def test_includes_page_number_when_present(self):
        tool = make_tool(doc=make_doc(), rows={"splitter": make_chunks(1)})
        r = await tool.run(document_id="doc-1")
        assert "第 1 页" in r.content

    @pytest.mark.asyncio
    async def test_human_readable_size(self):
        tool = make_tool(
            doc=make_doc(file_size=1536), rows={"splitter": make_chunks(1)}
        )
        r = await tool.run(document_id="doc-1")
        assert "1.5 KB" in r.content

    @pytest.mark.asyncio
    async def test_schema_shape(self):
        """供 LangGraph Agent 用的 OpenAI schema"""
        tool = make_tool()
        schema = tool.to_openai_schema()
        assert schema["function"]["name"] == "get_document_by_id"
        props = schema["function"]["parameters"]["properties"]
        assert set(props) == {"document_id", "offset", "limit"}
        assert schema["function"]["parameters"]["required"] == ["document_id"]


# ═══════════════════════════════════════════════════════════════
# MCP Server（内存 Client，不起子进程）
# ═══════════════════════════════════════════════════════════════

mcp_sdk = pytest.importorskip("mcp", reason="需要 mcp SDK")


class TestMcpServerSurface:
    """三层原语是否都正确注册"""

    @pytest.mark.asyncio
    async def test_lists_two_tools(self):
        from mcp import Client
        from app.mcp_server.server import create_server

        async with Client(create_server()) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            assert {"search_documents", "get_document_by_id"} <= names

    @pytest.mark.asyncio
    async def test_lists_resources(self):
        from mcp import Client
        from app.mcp_server.server import create_server

        async with Client(create_server()) as c:
            uris = {str(r.uri) for r in (await c.list_resources()).resources}
            assert "docs://list" in uris

    @pytest.mark.asyncio
    async def test_lists_prompts(self):
        from mcp import Client
        from app.mcp_server.server import create_server

        async with Client(create_server()) as c:
            names = {p.name for p in (await c.list_prompts()).prompts}
            assert "rag_qa" in names

    @pytest.mark.asyncio
    async def test_tool_schema_from_type_annotations(self):
        """MCP 的 schema 从类型注解生成（不是 Tool.parameters property）"""
        from mcp import Client
        from app.mcp_server.server import create_server

        async with Client(create_server()) as c:
            tools = {t.name: t for t in (await c.list_tools()).tools}
            schema = tools["get_document_by_id"].input_schema
            assert set(schema["properties"]) == {"document_id", "offset", "limit"}
            assert schema["required"] == ["document_id"]
            # 约束来自 Field(ge=..., le=...)
            assert schema["properties"]["limit"]["maximum"] == 50


class TestMcpErrorSemantics:
    """
    **MCP 最容易踩的坑**：handler 返回错误字符串时 `is_error=False`，
    模型会以为工具调用成功了、把那句话当成答案。
    必须 raise ToolError 才能让 `is_error=True`。
    """

    @pytest.mark.asyncio
    async def test_empty_query_is_error(self):
        from mcp import Client
        from app.mcp_server.server import create_server

        async with Client(create_server()) as c:
            r = await c.call_tool("search_documents", {"query": "   "})
            assert r.is_error is True

    @pytest.mark.asyncio
    async def test_unknown_document_is_error(self):
        from mcp import Client
        from app.mcp_server.server import create_server

        async with Client(create_server()) as c:
            r = await c.call_tool(
                "get_document_by_id",
                {"document_id": "00000000-0000-0000-0000-000000000000"},
            )
            assert r.is_error is True
            assert "未找到" in r.content[0].text

    @pytest.mark.asyncio
    async def test_invalid_prompt_args_handled(self):
        from mcp import Client
        from app.mcp_server.server import create_server

        async with Client(create_server()) as c:
            # 提示模板只是字符串拼接，任何输入都应正常返回
            r = await c.get_prompt("rag_qa", {"question": "什么是 RAG"})
            assert r.messages


class TestMcpPrompt:
    @pytest.mark.asyncio
    async def test_rag_qa_prompt_mentions_tools(self):
        """提示词应引导模型先用工具检索，体现三层原语的串联"""
        from mcp import Client
        from app.mcp_server.server import create_server

        async with Client(create_server()) as c:
            r = await c.get_prompt("rag_qa", {"question": "什么是 RAG"})
            text = r.messages[0].content.text
            assert "search_documents" in text
            assert "什么是 RAG" in text
