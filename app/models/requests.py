"""
请求体模型
定义各接口的请求体结构及参数校验
"""
from pydantic import BaseModel, Field


class PaginationParams(BaseModel):
    """分页查询参数"""
    page: int = Field(
        default=1,
        ge=1,
        description="页码（从1开始）",
        examples=[1],
    )
    page_size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="每页条数（最大100）",
        examples=[20],
    )


class DocumentSearchParams(PaginationParams):
    """文档搜索/过滤参数"""
    keyword: str | None = Field(
        default=None,
        description="按文件名模糊搜索",
        examples=["技术方案"],
    )
    status: str | None = Field(
        default=None,
        description="按状态过滤",
        examples=["uploaded", "ready"],
    )
    file_type: str | None = Field(
        default=None,
        description="按文件类型过滤",
        examples=[".pdf", ".txt"],
    )
    sort_by: str = Field(
        default="created_at",
        description="排序字段",
        examples=["created_at", "filename", "file_size"],
    )
    sort_order: str = Field(
        default="desc",
        pattern="^(asc|desc)$",
        description="排序方向：asc 升序 / desc 降序",
    )
