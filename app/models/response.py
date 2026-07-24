"""
统一API响应模型
所有接口使用统一的响应格式，便于前端统一处理
"""
from typing import Any, Generic, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """通用成功响应"""
    code: int = Field(default=200, description="业务状态码")
    message: str = Field(default="success", description="响应消息")
    data: T | None = Field(default=None, description="响应数据")

    model_config = {
        "json_schema_extra": {
            "example": {
                "code": 200,
                "message": "success",
                "data": {"id": "doc-a1b2c3d4", "filename": "技术方案.pdf"}
            }
        }
    }


class PaginatedData(BaseModel, Generic[T]):
    """分页数据结构"""
    items: list[T] = Field(default_factory=list, description="当前页数据")
    total: int = Field(default=0, description="总记录数")
    page: int = Field(default=1, description="当前页码")
    page_size: int = Field(default=20, description="每页条数")
    total_pages: int = Field(default=0, description="总页数")


class APIError(BaseModel):
    """错误响应"""
    code: int = Field(..., description="错误状态码")
    message: str = Field(..., description="错误消息")
    detail: str | None = Field(default=None, description="详细错误信息")

    model_config = {
        "json_schema_extra": {
            "example": {
                "code": 404,
                "message": "文档不存在",
                "detail": "ID 为 doc-xxx 的文档未找到"
            }
        }
    }
