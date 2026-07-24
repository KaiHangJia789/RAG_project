"""
文档管理接口测试
"""
import pytest


@pytest.fixture
async def uploaded_doc(async_client):
    """上传一个测试文档，返回 doc_id"""
    response = await async_client.post(
        "/api/v1/upload",
        files={"file": ("test.pdf", b"%PDF-1.4 mock content", "application/pdf")},
    )
    return response.json()["data"]["id"]


class TestListDocuments:
    """GET /api/v1/documents — 文档列表"""

    @pytest.mark.asyncio
    async def test_list_empty_returns_success(self, async_client):
        """无文档时列表应返回空数组"""
        response = await async_client.get("/api/v1/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["items"] == []
        assert data["data"]["total"] == 0
        assert data["data"]["page"] == 1
        assert data["data"]["total_pages"] >= 1

    @pytest.mark.asyncio
    async def test_list_with_documents(self, async_client, uploaded_doc):
        """上传文档后列表应有记录"""
        response = await async_client.get("/api/v1/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["total"] >= 1
        assert len(data["data"]["items"]) >= 1

    @pytest.mark.asyncio
    async def test_pagination_defaults(self, async_client):
        """默认分页参数应为 page=1, page_size=20"""
        response = await async_client.get("/api/v1/documents")
        data = response.json()
        assert data["data"]["page"] == 1
        assert data["data"]["page_size"] == 20

    @pytest.mark.asyncio
    async def test_pagination_custom(self, async_client):
        """自定义分页参数应生效"""
        response = await async_client.get("/api/v1/documents?page=1&page_size=5")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["page_size"] == 5

    @pytest.mark.asyncio
    async def test_page_must_be_positive(self, async_client):
        """page 参数 < 1 应返回 422"""
        response = await async_client.get("/api/v1/documents?page=0")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_page_size_max(self, async_client):
        """page_size > 100 应返回 422"""
        response = await async_client.get("/api/v1/documents?page_size=101")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_keyword_search(self, async_client):
        """关键词搜索应能匹配文件名"""
        # 先上传两个不同文件
        await async_client.post(
            "/api/v1/upload",
            files={"file": ("产品需求文档.pdf", b"%PDF-1.4", "application/pdf")},
        )
        await async_client.post(
            "/api/v1/upload",
            files={"file": ("技术方案.txt", b"hello", "text/plain")},
        )
        # 搜索 "产品"
        response = await async_client.get("/api/v1/documents?keyword=产品")
        data = response.json()
        assert data["data"]["total"] >= 1
        items = data["data"]["items"]
        filenames = [item["filename"] for item in items]
        assert any("产品" in f for f in filenames)

    @pytest.mark.asyncio
    async def test_status_filter(self, async_client, uploaded_doc):
        """状态过滤应返回正确结果"""
        response = await async_client.get("/api/v1/documents?status=uploaded")
        assert response.status_code == 200
        for item in response.json()["data"]["items"]:
            assert item["status"] == "uploaded"

    @pytest.mark.asyncio
    async def test_file_type_filter(self, async_client, uploaded_doc):
        """类型过滤应只返回指定类型"""
        response = await async_client.get("/api/v1/documents?file_type=.pdf")
        assert response.status_code == 200
        for item in response.json()["data"]["items"]:
            assert item["file_type"] == ".pdf"

    @pytest.mark.asyncio
    async def test_sort_order(self, async_client):
        """排序参数应生效"""
        await async_client.post(
            "/api/v1/upload",
            files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")},
        )
        await async_client.post(
            "/api/v1/upload",
            files={"file": ("b.pdf", b"%PDF-1.4", "application/pdf")},
        )
        # 按文件名升序
        resp_asc = await async_client.get(
            "/api/v1/documents?sort_by=filename&sort_order=asc"
        )
        items = resp_asc.json()["data"]["items"]
        if len(items) >= 2:
            assert items[0]["filename"] <= items[-1]["filename"]


class TestGetDocument:
    """GET /api/v1/documents/{id} — 文档详情"""

    @pytest.mark.asyncio
    async def test_get_existing_document(self, async_client, uploaded_doc):
        """获取已存在的文档应返回完整信息"""
        response = await async_client.get(f"/api/v1/documents/{uploaded_doc}")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["id"] == uploaded_doc
        assert data["data"]["filename"] == "test.pdf"
        assert "created_at" in data["data"]
        assert "updated_at" in data["data"]
        assert "storage_path" in data["data"]

    @pytest.mark.asyncio
    async def test_get_nonexistent_document(self, async_client):
        """获取不存在的文档应返回 404"""
        response = await async_client.get("/api/v1/documents/doc-nonexistent")
        assert response.status_code == 404
        data = response.json()
        assert "不存在" in data["message"]

    @pytest.mark.asyncio
    async def test_get_document_info(self, async_client, uploaded_doc):
        """获取文档元信息应返回轻量数据（不含 storage_path）"""
        response = await async_client.get(f"/api/v1/documents/{uploaded_doc}/info")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["id"] == uploaded_doc
        assert "storage_path" not in data["data"]
        # 应有基本字段
        for field in ["id", "filename", "file_type", "file_size", "status",
                       "created_at", "updated_at"]:
            assert field in data["data"], f"缺少字段: {field}"

    @pytest.mark.asyncio
    async def test_get_info_nonexistent(self, async_client):
        """获取不存在文档的元信息应返回 404"""
        response = await async_client.get("/api/v1/documents/doc-fake/info")
        assert response.status_code == 404


class TestDeleteDocument:
    """DELETE /api/v1/documents/{id} — 删除文档"""

    @pytest.mark.asyncio
    async def test_delete_existing_document(self, async_client, uploaded_doc):
        """删除已存在文档应成功"""
        response = await async_client.delete(f"/api/v1/documents/{uploaded_doc}")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["deleted"] is True
        assert data["data"]["id"] == uploaded_doc

    @pytest.mark.asyncio
    async def test_delete_nonexistent_document(self, async_client):
        """删除不存在文档应返回 404"""
        response = await async_client.delete("/api/v1/documents/doc-fake")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_deleted_document_not_accessible(self, async_client, uploaded_doc):
        """删除后再次访问应返回 404"""
        await async_client.delete(f"/api/v1/documents/{uploaded_doc}")
        response = await async_client.get(f"/api/v1/documents/{uploaded_doc}")
        assert response.status_code == 404


class TestSwaggerDocs:
    """OpenAPI 文档可访问性测试"""

    @pytest.mark.asyncio
    async def test_docs_page_accessible(self, async_client):
        """Swagger UI 页面应可访问"""
        response = await async_client.get("/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower() or "html" in str(response.headers.get("content-type", ""))

    @pytest.mark.asyncio
    async def test_openapi_json_accessible(self, async_client):
        """OpenAPI JSON Schema 应可访问"""
        response = await async_client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "RAG API"
        assert schema["info"]["version"] == "0.1.0"
        assert "paths" in schema
        # 验证所有 7 个端点都在 schema 中
        paths = schema["paths"]
        assert "/" in paths
        assert "/api/v1/health" in paths
        assert "/api/v1/health/ready" in paths
        assert "/api/v1/upload" in paths
        assert "/api/v1/documents" in paths
        assert "/api/v1/documents/{doc_id}" in paths
        assert "/api/v1/documents/{doc_id}/info" in paths

    @pytest.mark.asyncio
    async def test_redoc_page_accessible(self, async_client):
        """ReDoc 页面应可访问"""
        response = await async_client.get("/redoc")
        assert response.status_code == 200
