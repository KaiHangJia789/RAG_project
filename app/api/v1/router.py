"""
API v1 路由聚合
将各模块的路由统一注册到 v1 路由器下
"""
from fastapi import APIRouter

from app.api.v1.health import router as health_router
from app.api.v1.upload import router as upload_router
from app.api.v1.documents import router as documents_router

api_v1_router = APIRouter(prefix="/api/v1")

# 注册子路由
api_v1_router.include_router(health_router)
api_v1_router.include_router(upload_router)
api_v1_router.include_router(documents_router)
