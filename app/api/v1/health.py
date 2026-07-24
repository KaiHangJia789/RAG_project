"""
健康检查接口
提供应用存活性和就绪性检查端点
"""
from datetime import datetime, timezone
from fastapi import APIRouter

from app.config import settings
from app.models.response import APIResponse

router = APIRouter(tags=["健康检查"])


class HealthStatus:
    """健康状态常量"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@router.get(
    "/health",
    summary="健康检查",
    description="返回应用基本健康状态，可用于 Kubernetes liveness probe",
    response_model=APIResponse[dict],
)
async def health_check():
    """
    基础健康检查。
    返回应用名称、版本、当前时间和状态。
    """
    return APIResponse(
        code=200,
        message="服务运行正常",
        data={
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "status": HealthStatus.HEALTHY,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


@router.get(
    "/health/ready",
    summary="就绪检查",
    description="检查应用是否准备好接收流量（后续接入数据库检查）",
    response_model=APIResponse[dict],
)
async def readiness_check():
    """
    就绪检查（后续接入 DB/Redis 连接状态检查）。
    Kubernetes readiness probe 使用。
    """
    return APIResponse(
        code=200,
        message="应用就绪",
        data={
            "ready": True,
            "checks": {
                "api": "ok",
                "database": "skipped",  # Week 5-6 接入
                "redis": "skipped",     # Week 5-6 接入
            },
        },
    )
