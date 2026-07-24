"""
依赖注入模块
演示 FastAPI Depends 的用法：鉴权、分页参数、服务实例获取
"""
from typing import Annotated
from fastapi import Depends, Header, HTTPException, Query

from app.config import settings
from app.models.requests import PaginationParams
from app.services.document_service import DocumentService
from app.services.storage_service import StorageService

# ─── 单例服务实例 ──────────────────────────────────────────────
# 模块级别的单例，保证整个应用生命周期内只有一个实例
_storage_instance: StorageService | None = None
_document_service_instance: DocumentService | None = None


def get_storage_service() -> StorageService:
    """获取 StorageService 单例"""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = StorageService()
    return _storage_instance


def get_document_service() -> DocumentService:
    """获取 DocumentService 单例（依赖 StorageService）"""
    global _document_service_instance
    if _document_service_instance is None:
        _document_service_instance = DocumentService(get_storage_service())
    return _document_service_instance


# ─── 类型别名（方便路由函数签名） ───────────────────────────────
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]


# ─── 鉴权依赖（演示用） ────────────────────────────────────────
async def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str:
    """
    验证 API Key（简单演示版）
    实际项目中应接入 JWT 或 OAuth2

    Usage:
        @router.get("/secure")
        async def secure_endpoint(api_key: str = Depends(verify_api_key)):
            ...
    """
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="缺少 X-API-Key 请求头")
    # 简单演示：只要提供了 key 就通过
    # 生产环境应校验 key 的有效性（查数据库/Redis）
    return x_api_key


# ─── 分页参数依赖 ──────────────────────────────────────────────
async def pagination_params(
    page: int = Query(default=1, ge=1, description="页码（从1开始）"),
    page_size: int = Query(
        default=settings.DEFAULT_PAGE_SIZE,
        ge=1,
        le=settings.MAX_PAGE_SIZE,
        description=f"每页条数（最大{settings.MAX_PAGE_SIZE}）",
    ),
) -> PaginationParams:
    """将查询参数封装为 PaginationParams 对象"""
    return PaginationParams(page=page, page_size=page_size)
