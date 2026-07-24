"""
文件上传接口测试
"""
import pytest


class TestUploadDocument:
    """POST /api/v1/upload — 文档上传"""

    @pytest.mark.asyncio
    async def test_upload_pdf_success(self, async_client):
        """上传一个有效的 PDF 文件应成功"""
        response = await async_client.post(
            "/api/v1/upload",
            files={"file": ("test.pdf", b"%PDF-1.4 mock pdf content", "application/pdf")},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["code"] == 201
        assert data["message"] == "文档上传成功"
        assert data["data"]["filename"] == "test.pdf"
        assert data["data"]["file_type"] == ".pdf"
        assert data["data"]["status"] == "uploaded"
        assert data["data"]["id"].startswith("doc-")
        assert "created_at" in data["data"]

    @pytest.mark.asyncio
    async def test_upload_txt_success(self, async_client):
        """上传一个有效的 TXT 文件应成功"""
        content = "这是一段中文测试文本。\n第二行内容。"
        response = await async_client.post(
            "/api/v1/upload",
            files={"file": ("readme.txt", content.encode("utf-8"), "text/plain")},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["data"]["file_type"] == ".txt"
        assert data["data"]["file_size"] > 0

    @pytest.mark.asyncio
    async def test_upload_md_success(self, async_client):
        """上传 Markdown 文件应成功"""
        response = await async_client.post(
            "/api/v1/upload",
            files={"file": ("README.md", b"# Hello\nWorld", "text/markdown")},
        )
        assert response.status_code == 201
        assert response.json()["data"]["file_type"] == ".md"

    @pytest.mark.asyncio
    async def test_upload_csv_success(self, async_client):
        """上传 CSV 文件应成功"""
        response = await async_client.post(
            "/api/v1/upload",
            files={"file": ("data.csv", b"a,b,c\n1,2,3", "text/csv")},
        )
        assert response.status_code == 201
        assert response.json()["data"]["file_type"] == ".csv"

    @pytest.mark.asyncio
    async def test_upload_json_success(self, async_client):
        """上传 JSON 文件应成功"""
        response = await async_client.post(
            "/api/v1/upload",
            files={"file": ("config.json", b'{"key": "value"}', "application/json")},
        )
        assert response.status_code == 201
        assert response.json()["data"]["file_type"] == ".json"

    @pytest.mark.asyncio
    async def test_upload_rejects_unsupported_extension(self, async_client):
        """上传不支持的格式应返回 400"""
        response = await async_client.post(
            "/api/v1/upload",
            files={"file": ("image.png", b"fake png data", "image/png")},
        )
        assert response.status_code == 400
        data = response.json()
        assert data["code"] == 400
        assert "不支持" in data["message"] or "不支持" in str(data.get("detail", ""))

    @pytest.mark.asyncio
    async def test_upload_rejects_exe(self, async_client):
        """上传 .exe 文件应被拒绝"""
        response = await async_client.post(
            "/api/v1/upload",
            files={"file": ("malware.exe", b"\x00\x01\x02", "application/octet-stream")},
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_upload_without_file_returns_error(self, async_client):
        """不上传文件时应返回 422 校验错误"""
        response = await async_client.post("/api/v1/upload")
        assert response.status_code in (422, 400)

    @pytest.mark.asyncio
    async def test_upload_preserves_filename(self, async_client):
        """上传后返回的文件名应与原始文件名一致"""
        response = await async_client.post(
            "/api/v1/upload",
            files={"file": ("我的文档.PDF", b"%PDF-1.4 content", "application/pdf")},
        )
        assert response.status_code == 201
        assert response.json()["data"]["filename"] == "我的文档.PDF"

    @pytest.mark.asyncio
    async def test_upload_empty_file(self, async_client):
        """上传空文件应成功（但文件大小为0）"""
        response = await async_client.post(
            "/api/v1/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert response.status_code == 201
        assert response.json()["data"]["file_size"] == 0
