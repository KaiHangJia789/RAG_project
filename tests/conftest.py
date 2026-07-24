"""
Pytest 配置文件
提供测试客户端、fixtures 和共享工具函数
"""
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services.document_service import DocumentService
from app.services.storage_service import StorageService


@pytest.fixture
def storage_service(tmp_path):
    """创建使用临时目录的存储服务实例"""
    from app.config import settings
    # 临时覆盖上传目录
    original_upload_dir = settings.UPLOAD_DIR
    settings.UPLOAD_DIR = tmp_path / "uploads"
    service = StorageService()
    yield service
    settings.UPLOAD_DIR = original_upload_dir


@pytest.fixture
def document_service(storage_service):
    """创建使用临时存储的文档服务实例"""
    return DocumentService(storage_service)


@pytest.fixture
async def async_client():
    """创建异步 HTTP 测试客户端"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def sample_pdf_path(tmp_path):
    """创建一个示例 PDF 文件（用于上传测试）"""
    file_path = tmp_path / "test_sample.pdf"
    # 最小有效的 PDF 内容
    minimal_pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\n"
        b"startxref\n190\n%%EOF\n"
    )
    file_path.write_bytes(minimal_pdf)
    return file_path
