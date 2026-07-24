"""
请求耗时追踪中间件
对慢请求发出警告日志
"""
import logging
import time
from fastapi import Request

logger = logging.getLogger("rag_api.timing")

# 慢请求阈值（毫秒）
SLOW_REQUEST_THRESHOLD_MS = 1000


async def timing_middleware(request: Request, call_next):
    """
    耗时追踪中间件
    - 对所有请求计时
    - 超过阈值的请求输出 WARNING 级别日志
    """
    start = time.monotonic()
    response = await call_next(request)
    elapsed_ms = (time.monotonic() - start) * 1000

    if elapsed_ms > SLOW_REQUEST_THRESHOLD_MS:
        logger.warning(
            "慢请求警告 | %s %s | %.0fms (threshold=%dms)",
            request.method,
            request.url.path,
            elapsed_ms,
            SLOW_REQUEST_THRESHOLD_MS,
        )

    return response
