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
    async def test_list_returns_success(self, async_client):
        response = await async_client.get("/api/v1/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["page"] == 1
        assert data["data"]["page_size"] == 20

    @pytest.mark.asyncio
    async def test_list_with_documents(self, async_client, uploaded_doc):
        response = await async_client.get("/api/v1/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["total"] >= 1

    @pytest.mark.asyncio
    async def test_page_must_be_positive(self, async_client):
        response = await async_client.get("/api/v1/documents?page=0")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_page_size_max(self, async_client):
        response = await async_client.get("/api/v1/documents?page_size=101")
        assert response.status_code == 422


class TestGetDocument:
    """GET /api/v1/documents/{id} — 文档详情"""

    @pytest.mark.asyncio
    async def test_get_existing_document(self, async_client, uploaded_doc):
        response = await async_client.get(f"/api/v1/documents/{uploaded_doc}")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["id"] == uploaded_doc
        assert "created_at" in data["data"]

    @pytest.mark.asyncio
    async def test_get_nonexistent_document(self, async_client):
        response = await async_client.get("/api/v1/documents/doc-nonexistent")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_document_info(self, async_client, uploaded_doc):
        response = await async_client.get(f"/api/v1/documents/{uploaded_doc}/info")
        assert response.status_code == 200
        data = response.json()
        assert "storage_path" not in data["data"]

    @pytest.mark.asyncio
    async def test_get_info_nonexistent(self, async_client):
        response = await async_client.get("/api/v1/documents/doc-fake/info")
        assert response.status_code == 404


class TestDeleteDocument:
    """DELETE /api/v1/documents/{id} — 删除文档"""

    @pytest.mark.asyncio
    async def test_delete_existing_document(self, async_client, uploaded_doc):
        response = await async_client.delete(f"/api/v1/documents/{uploaded_doc}")
        assert response.status_code == 200
        assert response.json()["data"]["deleted"] is True

    @pytest.mark.asyncio
    async def test_delete_nonexistent_document(self, async_client):
        response = await async_client.delete("/api/v1/documents/doc-fake")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_deleted_document_not_accessible(self, async_client, uploaded_doc):
        await async_client.delete(f"/api/v1/documents/{uploaded_doc}")
        response = await async_client.get(f"/api/v1/documents/{uploaded_doc}")
        assert response.status_code == 404


class TestSwaggerDocs:
    """OpenAPI 文档可访问性测试"""

    @pytest.mark.asyncio
    async def test_docs_page_accessible(self, async_client):
        response = await async_client.get("/docs")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_openapi_json_accessible(self, async_client):
        response = await async_client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "RAG API"
        paths = schema["paths"]
        assert "/" in paths
        assert "/api/v1/health" in paths
        assert "/api/v1/upload" in paths
        assert "/api/v1/documents" in paths

    @pytest.mark.asyncio
    async def test_redoc_page_accessible(self, async_client):
        response = await async_client.get("/redoc")
        assert response.status_code == 200
