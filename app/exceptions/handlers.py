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


class UserNotFoundError(AppException):
    """
    默认用户不存在异常。

    典型场景：换库/重建库后没跑 seed.sql，users 表为空。
    返回 503 而非 500，并给出可操作的修复指令 —— 而不是让外键约束
    报一句 "documents_user_id_fkey 违反约束" 让人无从下手。
    """

    def __init__(self, username: str) -> None:
        super().__init__(
            message="服务未就绪",
            code=503,
            detail=(
                f"默认用户 '{username}' 不存在于 users 表。"
                f"请执行种子脚本初始化：psql -U <user> -d <db> -f app/db/migrations/seed.sql"
            ),
        )


class IndexNotReadyError(AppException):
    """
    向量索引未构建。

    返回 503 并给出构建命令 —— 索引缺失是个"跑一条命令就能修"的问题，
    不该让人对着"检索结果为空"猜半天。
    """

    def __init__(self, strategy: str | None = None) -> None:
        cmd = "python scripts/build_index.py"
        if strategy:
            cmd += f" --strategy {strategy}"
        super().__init__(
            message="知识库未就绪",
            code=503,
            detail=f"向量索引尚未构建。请先执行：{cmd}",
        )


class ChunkNotFoundError(AppException):
    """chunk 不存在（引用定位时）"""

    def __init__(self, chunk_id: str, hint: str | None = None) -> None:
        super().__init__(
            message="文本块不存在",
            code=404,
            detail=f"ID 为 '{chunk_id}' 的文本块未找到" + (f"（{hint}）" if hint else ""),
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
