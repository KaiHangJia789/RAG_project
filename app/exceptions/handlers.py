"""
全局异常处理器
统一捕获并格式化所有异常，返回一致的错误响应格式
"""
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.models.response import APIError


class AppException(Exception):
    """自定义应用异常基类"""

    def __init__(self, message: str, code: int = 400, detail: str | None = None) -> None:
        self.message = message
        self.code = code
        self.detail = detail
        super().__init__(message)


class DocumentNotFoundError(AppException):
    """文档不存在异常"""

    def __init__(self, doc_id: str) -> None:
        super().__init__(
            message="文档不存在",
            code=404,
            detail=f"ID 为 '{doc_id}' 的文档未找到",
        )


class FileValidationError(AppException):
    """文件校验失败异常"""

    def __init__(self, reason: str) -> None:
        super().__init__(
            message="文件校验失败",
            code=400,
            detail=reason,
        )


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """处理自定义应用异常"""
    return JSONResponse(
        status_code=exc.code,
        content=APIError(
            code=exc.code,
            message=exc.message,
            detail=exc.detail,
        ).model_dump(),
    )


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """处理标准HTTP异常"""
    return JSONResponse(
        status_code=exc.status_code,
        content=APIError(
            code=exc.status_code,
            message="请求错误" if exc.status_code < 500 else "服务器错误",
            detail=exc.detail,
        ).model_dump(),
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """处理 Pydantic 参数校验异常，返回清晰的校验错误信息"""
    errors: list[dict] = []
    for error in exc.errors():
        errors.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        })
    return JSONResponse(
        status_code=422,
        content=APIError(
            code=422,
            message="请求参数校验失败",
            detail=str(errors) if errors else None,
        ).model_dump(),
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """处理所有未捕获的异常（兜底）"""
    return JSONResponse(
        status_code=500,
        content=APIError(
            code=500,
            message="服务器内部错误",
            detail=str(exc) if isinstance(exc, Exception) else None,
        ).model_dump(),
    )
