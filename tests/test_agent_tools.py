"""
Agent 工具测试

重点是**安全边界** —— 工具参数是模型生成的文本，被提示注入诱导时
可能传入恶意内容。这些测试锁死各条防线。
"""
import pytest

from app.agent.tools import (
    CalculatorTool,
    DbQueryTool,
    LocalCorpusSearch,
    SearchDocsTool,
    ToolRegistry,
    UnsafeSqlError,
    WebSearchTool,
    build_default_registry,
    validate_readonly_sql,
)
from app.agent.tools.base import ToolResult
from app.services.index_service import IndexHit, IndexRecord


# ═══════════════════════════════════════════════════════════════
# 计算器
# ═══════════════════════════════════════════════════════════════

class TestCalculator:
    @pytest.mark.asyncio
    async def test_basic_arithmetic(self):
        reg = ToolRegistry([CalculatorTool()])
        r = await reg.execute("calculator", '{"expression": "123 * 456"}')
        assert r.ok and "56088" in r.content

    @pytest.mark.asyncio
    async def test_operator_precedence(self):
        reg = ToolRegistry([CalculatorTool()])
        r = await reg.execute("calculator", '{"expression": "(2 + 3) * 4"}')
        assert r.ok and "20" in r.content

    @pytest.mark.asyncio
    async def test_allowed_functions(self):
        reg = ToolRegistry([CalculatorTool()])
        assert (await reg.execute("calculator", '{"expression": "sqrt(16)"}')).ok
        assert (await reg.execute("calculator", '{"expression": "max(1, 9, 5)"}')).ok
        assert (await reg.execute("calculator", '{"expression": "round(3.14159, 2)"}')).ok

    @pytest.mark.asyncio
    async def test_integer_result_has_no_decimal(self):
        """56088.0 要显示成 56088（避免给模型无意义的浮点尾巴）"""
        reg = ToolRegistry([CalculatorTool()])
        r = await reg.execute("calculator", '{"expression": "100 * 2"}')
        assert "200.0" not in r.content
        assert "200" in r.content

    @pytest.mark.asyncio
    async def test_division_by_zero_handled(self):
        reg = ToolRegistry([CalculatorTool()])
        r = await reg.execute("calculator", '{"expression": "1/0"}')
        assert not r.ok and "零" in r.content

    @pytest.mark.asyncio
    async def test_empty_expression(self):
        reg = ToolRegistry([CalculatorTool()])
        r = await reg.execute("calculator", "{}")
        assert not r.ok


class TestCalculatorSecurity:
    """
    安全红线：**绝不用 eval**。

    以下表达式如果走 eval 会直接执行任意代码。它们必须全部被拒绝。
    """

    @pytest.mark.parametrize("malicious", [
        '__import__("os").system("whoami")',
        'open("/etc/passwd").read()',
        'eval("1+1")',
        'exec("x=1")',
        '__builtins__',
        'globals()',
        '1 if True else 2',
        '(lambda: 1)()',
        '[x for x in range(3)]',
    ])
    @pytest.mark.asyncio
    async def test_rejects_code_execution(self, malicious):
        import json

        reg = ToolRegistry([CalculatorTool()])
        r = await reg.execute("calculator", json.dumps({"expression": malicious}))
        assert not r.ok, f"必须拒绝: {malicious}"

    @pytest.mark.asyncio
    async def test_rejects_overlong_expression(self):
        reg = ToolRegistry([CalculatorTool()])
        r = await reg.execute("calculator", '{"expression": "' + "1+" * 400 + '1"}')
        assert not r.ok and ("过长" in r.content or "不被允许" in r.content)

    @pytest.mark.asyncio
    async def test_huge_exponent_does_not_hang(self):
        """巨大的幂运算可能耗尽 CPU，应被表达式长度限制挡住或正常返回"""
        reg = ToolRegistry([CalculatorTool()])
        r = await reg.execute("calculator", '{"expression": "9**9**9"}')
        # 不断言成败，只断言没有崩溃
        assert isinstance(r, ToolResult)


# ═══════════════════════════════════════════════════════════════
# 参数解析容错
# ═══════════════════════════════════════════════════════════════

class TestArgumentParsing:
    @pytest.mark.asyncio
    async def test_invalid_json_returns_error_not_exception(self):
        reg = ToolRegistry([CalculatorTool()])
        r = await reg.execute("calculator", "这不是 JSON")
        assert not r.ok and "JSON" in r.content

    @pytest.mark.asyncio
    async def test_empty_arguments_treated_as_no_params(self):
        """无参数工具会收到空串"""
        reg = ToolRegistry([CalculatorTool()])
        r = await reg.execute("calculator", "")
        assert not r.ok        # 缺 expression，但不应崩
        assert "表达式为空" in r.content

    @pytest.mark.asyncio
    async def test_dict_arguments_accepted(self):
        """部分 SDK 版本直接给 dict 而非 JSON 字符串"""
        reg = ToolRegistry([CalculatorTool()])
        r = await reg.execute("calculator", {"expression": "1+1"})
        assert r.ok

    @pytest.mark.asyncio
    async def test_unknown_tool_lists_available(self):
        reg = ToolRegistry([CalculatorTool()])
        r = await reg.execute("nonexistent", "{}")
        assert not r.ok and "calculator" in r.content

    @pytest.mark.asyncio
    async def test_unexpected_kwargs_handled(self):
        """模型可能给 schema 之外的字段"""
        reg = ToolRegistry([CalculatorTool()])
        r = await reg.execute("calculator", '{"expression": "1+1", "bogus": 1}')
        assert isinstance(r, ToolResult)   # 不崩即可


# ═══════════════════════════════════════════════════════════════
# SQL 安全
# ═══════════════════════════════════════════════════════════════

class TestSqlSafety:
    @pytest.mark.parametrize("sql", [
        "SELECT filename FROM documents",
        "SELECT id, chunk_index FROM chunks WHERE document_id = 'x'",
        'SELECT d.filename FROM "documents" d',
    ])
    def test_allows_readonly_whitelisted(self, sql):
        validate_readonly_sql(sql)      # 不抛异常即通过

    @pytest.mark.parametrize("sql", [
        "DROP TABLE users",
        "DELETE FROM documents",
        "UPDATE documents SET status='x'",
        "INSERT INTO documents VALUES (1)",
        "TRUNCATE chunks",
        "ALTER TABLE documents ADD COLUMN x int",
    ])
    def test_rejects_write_operations(self, sql):
        with pytest.raises(UnsafeSqlError):
            validate_readonly_sql(sql)

    @pytest.mark.parametrize("sql", [
        "SELECT * FROM users",                    # 敏感表
        "SELECT password_hash FROM users",
        "SELECT id FROM documents; DROP TABLE users",   # 多语句
        "SELECT 1",                                # 未识别表
    ])
    def test_rejects_unsafe_targets(self, sql):
        with pytest.raises(UnsafeSqlError):
            validate_readonly_sql(sql)

    def test_string_literal_containing_keyword_is_ok(self):
        """'DROP' 出现在字符串字面量里不应误判"""
        validate_readonly_sql(
            "SELECT filename FROM documents WHERE filename = 'DROP TABLE'"
        )


class TestDbQueryTool:
    class FakeDb:
        def __init__(self, rows=None, one=None):
            self._rows = rows or []
            self._one = one

        async def fetch_all(self, q, *args):
            return self._rows

        async def fetch_one(self, q, *args):
            return self._one

    @pytest.mark.asyncio
    async def test_list_documents(self):
        db = self.FakeDb(rows=[
            {"filename": "a.md", "file_type": ".md", "file_size": 100, "status": "ready"},
            {"filename": "b.pdf", "file_type": ".pdf", "file_size": 200, "status": "ready"},
        ])
        r = await DbQueryTool(db).run(action="list_documents")
        assert r.ok and "2 篇" in r.content

    @pytest.mark.asyncio
    async def test_count_chunks_requires_filename(self):
        r = await DbQueryTool(self.FakeDb()).run(action="count_chunks")
        assert not r.ok and "filename" in r.content

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        r = await DbQueryTool(self.FakeDb()).run(action="drop_everything")
        assert not r.ok and "未知动作" in r.content


# ═══════════════════════════════════════════════════════════════
# 检索工具
# ═══════════════════════════════════════════════════════════════

def make_hit(fid: int, text: str, score: float, filename="doc.md", page=None):
    return IndexHit(
        faiss_id=fid, score=score,
        record=IndexRecord(
            faiss_id=fid, chunk_id=f"c{fid}", document_id="d1",
            filename=filename, chunk_index=fid, text=text, page_number=page,
        ),
    )


class FakeIndex:
    def __init__(self, hits=None, ready=True, records=None):
        self._hits = hits or []
        self._ready = ready
        self._records = records or []
        self.queries: list[str] = []

    @property
    def ready(self):
        return self._ready

    async def search(self, query, **kwargs):
        self.queries.append(query)
        return self._hits

    def all_records(self):
        return self._records


class TestSearchDocsTool:
    @pytest.mark.asyncio
    async def test_returns_formatted_hits(self):
        idx = FakeIndex([make_hit(1, "RAG 是检索增强生成", 0.85, "rag.md", 2)])
        r = await SearchDocsTool(idx).run(query="什么是 RAG")
        assert r.ok
        assert "rag.md" in r.content and "第2页" in r.content

    @pytest.mark.asyncio
    async def test_exposes_raw_hits_in_meta(self):
        """
        原始命中必须放在 meta.raw_hits —— 上层靠它做引用映射，
        而不是用关键词二次回捞（那样不准且脆弱）。
        """
        hits = [make_hit(1, "内容", 0.8)]
        r = await SearchDocsTool(FakeIndex(hits)).run(query="查询")
        assert r.meta.get("raw_hits") == hits

    @pytest.mark.asyncio
    async def test_index_not_ready(self):
        r = await SearchDocsTool(FakeIndex(ready=False)).run(query="查询")
        assert not r.ok and "尚未就绪" in r.content

    @pytest.mark.asyncio
    async def test_empty_query_rejected(self):
        r = await SearchDocsTool(FakeIndex()).run(query="   ")
        assert not r.ok

    @pytest.mark.asyncio
    async def test_top_k_clamped(self):
        """模型可能给 0、负数或超大值"""
        idx = FakeIndex([make_hit(1, "x", 0.8)])
        tool = SearchDocsTool(idx)
        await tool.run(query="q", top_k=0)
        await tool.run(query="q", top_k=-5)
        await tool.run(query="q", top_k=9999)
        assert all(c["top_k"] <= 20 for c in tool.calls)

    @pytest.mark.asyncio
    async def test_no_hits_message(self):
        r = await SearchDocsTool(FakeIndex([])).run(query="无关问题")
        assert r.ok and "没有找到" in r.content


class TestLocalCorpusSearch:
    @pytest.mark.asyncio
    async def test_keyword_matching(self):
        recs = [
            make_hit(1, "Keyset 分页使用游标", 0.9, "pg.md").record,
            make_hit(2, "Redis 缓存策略", 0.8, "redis.md").record,
        ]
        backend = LocalCorpusSearch(FakeIndex(records=recs))
        results = await backend.search("Keyset 分页")
        assert results and "pg.md" in results[0]["title"]

    @pytest.mark.asyncio
    async def test_no_match_returns_empty(self):
        backend = LocalCorpusSearch(FakeIndex(records=[]))
        assert await backend.search("完全无关") == []


class TestWebSearchTool:
    @pytest.mark.asyncio
    async def test_pluggable_backend(self):
        """换成任何实现了 SearchBackend 协议的后端都能工作"""

        class FakeBackend:
            async def search(self, query, *, max_results=5):
                return [{"title": "T1", "snippet": "S1", "url": "u1"}]

        r = await WebSearchTool(FakeBackend()).run(query="查询")
        assert r.ok and "T1" in r.content

    @pytest.mark.asyncio
    async def test_empty_results(self):
        class EmptyBackend:
            async def search(self, query, *, max_results=5):
                return []

        r = await WebSearchTool(EmptyBackend()).run(query="查询")
        assert r.ok and "没有找到" in r.content


# ═══════════════════════════════════════════════════════════════
# 注册表组装
# ═══════════════════════════════════════════════════════════════

class TestRegistryFactory:
    def test_calculator_always_present(self):
        reg = build_default_registry()
        assert "calculator" in reg.names()

    def test_search_tools_require_ready_index(self):
        """索引未就绪时不注册检索工具 —— 避免模型把轮次浪费在注定失败的工具上"""
        assert "search_documents" not in build_default_registry(
            index_service=FakeIndex(ready=False)
        ).names()

    def test_search_tools_registered_when_ready(self):
        names = build_default_registry(index_service=FakeIndex(ready=True)).names()
        assert "search_documents" in names and "web_search" in names

    def test_db_tool_requires_db(self):
        assert "query_metadata" not in build_default_registry().names()
        assert "query_metadata" in build_default_registry(db=object()).names()

    def test_openai_schema_shape(self):
        """导出的 schema 必须符合 OpenAI function calling 格式"""
        reg = ToolRegistry([CalculatorTool()])
        schema = reg.to_openai_schema()
        assert len(schema) == 1
        fn = schema[0]
        assert fn["type"] == "function"
        assert fn["function"]["name"] == "calculator"
        assert "parameters" in fn["function"]
        assert fn["function"]["parameters"]["type"] == "object"
