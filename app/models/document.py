"""
文档数据模型
定义文档的 Pydantic 模型，用于序列化、反序列化和参数校验
"""
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class DocumentStatus(str, Enum):
    """文档处理状态"""
    UPLOADED = "uploaded"
    PARSING = "parsing"
    CHUNKING = "chunking"
    READY = "ready"
    FAILED = "failed"


class DocumentBase(BaseModel):
    """文档基础字段"""
    filename: str = Field(..., description="原始文件名", examples=["技术方案.pdf"])
    file_type: str = Field(..., description="文件扩展名（含点）", examples=[".pdf", ".txt", ".md"])
    file_size: int = Field(..., ge=0, description="文件大小（字节）", examples=[102400])


class DocumentCreate(DocumentBase):
    """创建文档时需要的字段（由上传接口内部使用）"""
    storage_path: str = Field(..., description="服务器存储路径")


class DocumentResponse(DocumentBase):
    """返回给客户端的文档完整信息"""
    id: str = Field(..., description="文档唯一ID", examples=["doc-a1b2c3d4"])
    status: DocumentStatus = Field(default=DocumentStatus.UPLOADED, description="处理状态")
    storage_path: str = Field(..., description="服务器存储路径")
    created_at: datetime = Field(..., description="创建时间 (UTC)")
    updated_at: datetime = Field(..., description="最后更新时间 (UTC)")

    # Pydantic v2 配置
    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "doc-a1b2c3d4",
                "filename": "技术方案.pdf",
                "file_type": ".pdf",
                "file_size": 204800,
                "status": "uploaded",
                "storage_path": "uploads/2026/08/doc-a1b2c3d4.pdf",
                "created_at": "2026-08-03T10:30:00Z",
                "updated_at": "2026-08-03T10:30:00Z",
            }
        }
    }


class DocumentInfo(BaseModel):
    """文档元信息（轻量版，不包含存储路径等内部字段）"""
    id: str
    filename: str
    file_type: str
    file_size: int
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime
