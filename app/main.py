"""
FastAPI 应用入口
负责：
- 创建 FastAPI 实例
- 注册中间件
- 注册全局异常处理器
- 注册路由
- 管理应用生命周期（startup/shutdown）
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import api_v1_router
from app.config import settings
from app.exceptions.handlers import (
    AppException,
    app_exception_handler,
    general_exception_handler,
    http_exception_handler,
    validation_exception_handler,
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


# ─── 生命周期管理 ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    - startup: 初始化资源（创建目录、预热连接池等）
    - shutdown: 清理资源（关闭连接、刷新缓冲区等）
    """
    # ===== Startup =====
    logger.info("=" * 60)
    logger.info("🚀 %s v%s 启动中...", settings.APP_NAME, settings.APP_VERSION)
    logger.info("   环境: %s", "DEVELOPMENT" if settings.DEBUG else "PRODUCTION")
    logger.info("   上传目录: %s", settings.UPLOAD_DIR.absolute())

    # 确保上传目录存在
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("✅ 应用启动完成")
    logger.info("=" * 60)

    yield  # ← 应用运行期间

    # ===== Shutdown =====
    logger.info("=" * 60)
    logger.info("🛑 %s 正在关闭...", settings.APP_NAME)
    logger.info("✅ 应用已安全关闭")
    logger.info("=" * 60)


# ─── 创建 FastAPI 应用 ─────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    lifespan=lifespan,
    # Swagger UI 配置
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    # OpenAPI 文档的额外元信息
    swagger_ui_parameters={
        "defaultModelsExpandDepth": 1,
        "docExpansion": "list",
        "filter": True,
    },
)


# ─── 注册中间件（后注册的先执行） ───────────────────────────────
app.middleware("http")(timing_middleware)
app.middleware("http")(log_request_middleware)


# ─── 注册全局异常处理器 ─────────────────────────────────────────
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)


# ─── 注册路由 ──────────────────────────────────────────────────
app.include_router(api_v1_router)


# ─── 根路由 ────────────────────────────────────────────────────
@app.get(
    "/",
    summary="API 根信息",
    description="返回 API 的基本信息、版本和可用接口列表",
    tags=["系统"],
)
async def root():
    """API 根路径，返回服务概览"""
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
            "文件上传": {
                "POST /api/v1/upload": "上传文档",
            },
            "文档管理": {
                "GET /api/v1/documents": "文档列表（分页+搜索+过滤+排序）",
                "GET /api/v1/documents/{id}": "获取文档详情",
                "GET /api/v1/documents/{id}/info": "获取文档元信息",
                "DELETE /api/v1/documents/{id}": "删除文档",
            },
        },
    }
