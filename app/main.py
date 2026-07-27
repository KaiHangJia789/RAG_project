"""
FastAPI 应用入口
Week5: 新增 PostgreSQL + Redis 生命周期管理
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import api_v1_router
from app.cache.connection import RedisClient
from app.config import settings
from app.db.connection import Database
from app.dependencies.auth import init_services
from app.exceptions.handlers import (
    AppException,
    app_exception_handler,
    general_exception_handler,
)
from app.middleware.logging import log_request_middleware
from app.middleware.timing import timing_middleware

# ─── 日志配置 ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag_api")

# ─── 全局资源句柄 ──────────────────────────────────────────────
db = Database()
redis_client = RedisClient()


# ─── 生命周期管理 ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    - startup: 初始化 PostgreSQL 连接池 + Redis 连接 + Service 单例
    - shutdown: 关闭所有连接
    """
    # ===== Startup =====
    logger.info("=" * 60)
    logger.info("🚀 %s v%s 启动中...", settings.APP_NAME, settings.APP_VERSION)

    await db.connect(settings.DATABASE_DSN)
    await redis_client.connect(settings.REDIS_URL)
    init_services(db, redis_client)

    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("   PostgreSQL: %s:%s/%s", settings.POSTGRES_HOST, settings.POSTGRES_PORT, settings.POSTGRES_DB)
    logger.info("   Redis:      %s:%s/%s", settings.REDIS_HOST, settings.REDIS_PORT, settings.REDIS_DB)
    logger.info("   上传目录:   %s", settings.UPLOAD_DIR.absolute())
    logger.info("✅ 应用启动完成")
    logger.info("=" * 60)

    yield  # ← 应用运行期间

    # ===== Shutdown =====
    logger.info("=" * 60)
    logger.info("🛑 %s 正在关闭...", settings.APP_NAME)
    await redis_client.disconnect()
    await db.disconnect()
    logger.info("✅ 应用已安全关闭")
    logger.info("=" * 60)


# ─── 创建 FastAPI 应用 ─────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    swagger_ui_parameters={
        "defaultModelsExpandDepth": 1,
        "docExpansion": "list",
        "filter": True,
    },
)

# ─── 注册中间件 ────────────────────────────────────────────────
app.middleware("http")(timing_middleware)
app.middleware("http")(log_request_middleware)

# ─── 注册异常处理器 ────────────────────────────────────────────
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# ─── 注册路由 ──────────────────────────────────────────────────
app.include_router(api_v1_router)


# ─── 根路由 ────────────────────────────────────────────────────
@app.get(
    "/",
    summary="API 根信息",
    tags=["系统"],
)
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": settings.APP_DESCRIPTION,
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "健康检查": {
                "GET /health": "基础健康检查",
                "GET /health/ready": "就绪检查",
            },
            "文件上传": {"POST /api/v1/upload": "上传文档"},
            "文档管理": {
                "GET /api/v1/documents": "文档列表（Keyset分页+搜索+过滤）",
                "GET /api/v1/documents/{id}": "获取文档详情（Cache-Aside）",
                "GET /api/v1/documents/{id}/info": "获取文档元信息",
                "DELETE /api/v1/documents/{id}": "删除文档",
            },
        },
    }
