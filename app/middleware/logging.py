"""
请求日志中间件
记录每个 HTTP 请求的方法、路径、状态码和处理时间
"""
import logging
import time
from fastapi import Request

logger = logging.getLogger("rag_api")


async def log_request_middleware(request: Request, call_next):
    """
    记录请求日志的中间件
    - 请求到达时记录 method + path
    - 响应返回时记录 status + 耗时
    """
    start_time = time.monotonic()

    logger.info(
        "→ %s %s | client=%s",
        request.method,
        request.url.path,
        request.client.host if request.client else "unknown",
    )

    response = await call_next(request)

    elapsed_ms = (time.monotonic() - start_time) * 1000

    logger.info(
        "← %s %s | status=%d | %.2fms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )

    # 在响应头中添加处理时间
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response
