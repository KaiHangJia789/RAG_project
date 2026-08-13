"""
依赖注入模块
提供 FastAPI Depends 依赖：DB、Redis、Service、鉴权
"""
from typing import Annotated
from fastapi import Depends, Header, HTTPException, Query

from app.config import settings
from app.models.requests import PaginationParams

# ── 全局单例（由 main.py lifespan 初始化后赋值） ──
from app.db.connection import Database
from app.cache.connection import RedisClient
from app.cache.document_cache import DocumentCache
from app.services.document_service import DocumentService
from app.services.parsing_service import ParsingService
from app.services.storage_service import StorageService

_db: Database | None = None
_redis_client: RedisClient | None = None
_doc_cache: DocumentCache | None = None
_doc_service: DocumentService | None = None
_parsing_service: ParsingService | None = None


def init_services(db: Database, redis_client: RedisClient) -> None:
    """在 app startup 时调用，初始化所有服务单例"""
    global _db, _redis_client, _doc_cache, _doc_service, _parsing_service
    _db = db
    _redis_client = redis_client
    _doc_cache = DocumentCache(redis_client.client)

    storage = StorageService()
    _doc_service = DocumentService(
        db=db,
        cache=_doc_cache,
        storage=storage,
    )
    _parsing_service = ParsingService(db=db)


# ── 依赖获取函数 ──────────────────────────────────────────────

def get_db() -> Database:
    if _db is None:
        raise HTTPException(status_code=503, detail="数据库未连接，请稍后重试")
    return _db


def get_document_service() -> DocumentService:
    if _doc_service is None:
        # DB 未就绪时优雅降级：返回 503 而非 500，前端能明确感知服务不可用
        raise HTTPException(status_code=503, detail="数据库未连接，文档服务暂不可用")
    return _doc_service


def get_parsing_service() -> ParsingService:
    if _parsing_service is None:
        raise HTTPException(status_code=503, detail="数据库未连接，解析服务暂不可用")
    return _parsing_service


# ── 类型别名（方便路由函数签名） ───────────────────────────────
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
ParsingServiceDep = Annotated[ParsingService, Depends(get_parsing_service)]


# ── 鉴权依赖（演示用 — Week5 只校验 Header 存在） ──────────────
async def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str:
    """验证 API Key（简单演示版）"""
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="缺少 X-API-Key 请求头")
    return x_api_key


# ── 分页参数依赖 ──────────────────────────────────────────────
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
