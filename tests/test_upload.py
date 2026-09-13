"""
文件上传接口测试
"""
import pytest

from app.config import settings


class TestUploadDocument:
    """POST /api/v1/upload — 文档上传"""

    @pytest.mark.asyncio
    async def test_upload_pdf_success(self, async_client):
        """上传 PDF — 解析可能失败但文档应存储成功"""
        # 注意：mock PDF 内容不是真正的 PDF，pdfplumber 解析会失败
        # 但上传流程不会因解析失败而中断，只是状态标记为 'failed'
        response = await async_client.post(
            "/api/v1/upload",
            files={"file": ("test.pdf", b"%PDF-1.4 mock pdf content", "application/pdf")},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["data"]["filename"] == "test.pdf"
        assert data["data"]["file_type"] == ".pdf"
        assert data["data"]["status"] in ("uploaded", "ready", "failed")

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


class TestDefaultUserResolution:
    """
    回归测试：documents.user_id 有指向 users 表的外键约束。

    历史 bug：代码里把 user_id 写成 UUID 常量。那个 UUID 是云服务器库里的
    demo_user；迁移到本机 PostgreSQL 后 seed.sql 用 gen_random_uuid()
    重新生成了新 UUID，常量随即失效 → 每次上传都撞
    documents_user_id_fkey 外键约束 → 500「服务器内部错误」，
    而文件在报错前已落盘 → 磁盘上堆孤儿文件、列表页永远查不到记录。

    现在改为按 username 查 users 表（环境无关），失败时返回可诊断的 503。
    """

    @pytest.mark.asyncio
    async def test_upload_binds_to_user_row_not_hardcoded_uuid(
        self, async_client, fake_db
    ):
        """上传的文档 user_id 必须等于 users 表里 demo_user 的真实 id"""
        response = await async_client.post(
            "/api/v1/upload",
            files={"file": ("fk_check.txt", b"hello fk", "text/plain")},
        )
        assert response.status_code == 201

        doc_id = response.json()["data"]["id"]
        assert fake_db.store[doc_id]["user_id"] == fake_db.user_id("demo_user")

    @pytest.mark.asyncio
    async def test_missing_user_returns_503_with_actionable_detail(
        self, async_client, fake_db
    ):
        """users 表没有该用户时返回 503 + 修复指引，而不是外键约束导致的 500"""
        fake_db.users.clear()

        response = await async_client.post(
            "/api/v1/upload",
            files={"file": ("no_user.txt", b"hello", "text/plain")},
        )
        assert response.status_code == 503

        body = response.json()
        assert "seed.sql" in (body.get("detail") or "")
        assert settings.DEFAULT_USERNAME in (body.get("detail") or "")

    @pytest.mark.asyncio
    async def test_missing_user_leaves_no_orphan_file(
        self, async_client, fake_db, storage_service
    ):
        """默认用户不存在时，不能把文件落到磁盘上（避免孤儿文件）"""
        fake_db.users.clear()

        await async_client.post(
            "/api/v1/upload",
            files={"file": ("orphan.txt", b"orphan check", "text/plain")},
        )

        leftovers = list(settings.UPLOAD_DIR.rglob("*.txt"))
        assert leftovers == [], f"遗留孤儿文件: {leftovers}"

    @pytest.mark.asyncio
    async def test_uploaded_document_appears_in_list(self, async_client):
        """上传和列表用同一个 user_id —— 上传后立刻能在列表里查到"""
        await async_client.post(
            "/api/v1/upload",
            files={"file": ("listed.txt", b"listed content", "text/plain")},
        )

        response = await async_client.get("/api/v1/documents")
        assert response.status_code == 200

        filenames = [i["filename"] for i in response.json()["data"]["items"]]
        assert "listed.txt" in filenames

    @pytest.mark.asyncio
    async def test_status_persisted_after_parse(self, async_client, fake_db):
        """解析完成后状态要落库，列表页才能看到 ready/failed 而不是 uploaded"""
        response = await async_client.post(
            "/api/v1/upload",
            files={"file": ("status.txt", b"status check", "text/plain")},
        )
        doc_id = response.json()["data"]["id"]

        # 响应里的状态与 DB 中的状态必须一致（此前只改了内存对象，DB 恒为 uploaded）
        assert fake_db.store[doc_id]["status"] == response.json()["data"]["status"]
