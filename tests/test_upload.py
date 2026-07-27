"""
文件上传接口测试
"""
import pytest


class TestUploadDocument:
    """POST /api/v1/upload — 文档上传"""

    @pytest.mark.asyncio
    async def test_upload_pdf_success(self, async_client):
        response = await async_client.post(
            "/api/v1/upload",
            files={"file": ("test.pdf", b"%PDF-1.4 mock pdf content", "application/pdf")},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["data"]["filename"] == "test.pdf"
        assert data["data"]["file_type"] == ".pdf"
        assert data["data"]["status"] == "uploaded"

    @pytest.mark.asyncio
    async def test_upload_txt_success(self, async_client):
        content = "中文测试文本。\n第二行内容。"
        response = await async_client.post(
            "/api/v1/upload",
            files={"file": ("readme.txt", content.encode("utf-8"), "text/plain")},
        )
        assert response.status_code == 201
        assert response.json()["data"]["file_type"] == ".txt"

    @pytest.mark.asyncio
    async def test_upload_rejects_unsupported_extension(self, async_client):
        response = await async_client.post(
            "/api/v1/upload",
            files={"file": ("image.png", b"fake png data", "image/png")},
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_upload_rejects_exe(self, async_client):
        response = await async_client.post(
            "/api/v1/upload",
            files={"file": ("malware.exe", b"\x00\x01\x02", "application/octet-stream")},
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_upload_without_file_returns_error(self, async_client):
        response = await async_client.post("/api/v1/upload")
        assert response.status_code in (422, 400)
