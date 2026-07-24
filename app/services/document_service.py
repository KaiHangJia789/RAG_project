"""
文档业务逻辑服务
负责文档的CRUD操作、查询过滤、分页等
"""
import uuid
from datetime import datetime, UTC
from typing import Optional

from app.config import settings
from app.models.document import (
    DocumentCreate,
    DocumentInfo,
    DocumentResponse,
    DocumentStatus,
)
from app.models.requests import DocumentSearchParams
from app.models.response import PaginatedData
from app.services.storage_service import StorageService


class DocumentService:
    """
    文档服务（当前为内存存储实现）
    Week 5-6 将替换为 PostgreSQL + Redis 实现
    """

    def __init__(self, storage: StorageService) -> None:
        self._storage = storage
        self._documents: dict[str, DocumentResponse] = {}  # 内存存储

    def _generate_id(self) -> str:
        """生成唯一文档ID"""
        return f"doc-{uuid.uuid4().hex[:12]}"

    async def create_document(
        self, filename: str, content: bytes
    ) -> DocumentResponse:
        """上传并创建文档记录"""
        # 1. 文件校验
        self._storage.validate_file(filename, len(content))

        # 2. 物理存储
        storage_result = await self._storage.save(filename, content)

        # 3. 创建记录
        now = datetime.now(UTC)
        doc = DocumentResponse(
            id=self._generate_id(),
            filename=filename,
            file_type=storage_result["file_type"],
            file_size=storage_result["file_size"],
            status=DocumentStatus.UPLOADED,
            storage_path=storage_result["storage_path"],
            created_at=now,
            updated_at=now,
        )
        self._documents[doc.id] = doc
        return doc

    def get_document(self, doc_id: str) -> DocumentResponse | None:
        """获取文档详情"""
        return self._documents.get(doc_id)

    def search_documents(self, params: DocumentSearchParams) -> PaginatedData:
        """搜索文档列表（支持过滤、排序、分页）"""
        docs = list(self._documents.values())

        # 过滤
        if params.keyword:
            keyword_lower = params.keyword.lower()
            docs = [d for d in docs if keyword_lower in d.filename.lower()]
        if params.status:
            docs = [d for d in docs if d.status.value == params.status]
        if params.file_type:
            docs = [d for d in docs if d.file_type == params.file_type]

        # 排序
        reverse = params.sort_order == "desc"
        docs.sort(key=lambda d: getattr(d, params.sort_by), reverse=reverse)

        # 分页
        total = len(docs)
        total_pages = max(1, (total + params.page_size - 1) // params.page_size)
        start = (params.page - 1) * params.page_size
        end = start + params.page_size
        page_items = docs[start:end]

        return PaginatedData(
            items=page_items,
            total=total,
            page=params.page,
            page_size=params.page_size,
            total_pages=total_pages,
        )

    def delete_document(self, doc_id: str) -> bool:
        """删除文档记录及物理文件"""
        doc = self._documents.get(doc_id)
        if doc is None:
            return False
        # 删除物理文件
        self._storage.delete(doc.storage_path)
        # 删除记录
        del self._documents[doc_id]
        return True

    def get_document_info(self, doc_id: str) -> DocumentInfo | None:
        """获取文档元信息（轻量版）"""
        doc = self._documents.get(doc_id)
        if doc is None:
            return None
        return DocumentInfo(
            id=doc.id,
            filename=doc.filename,
            file_type=doc.file_type,
            file_size=doc.file_size,
            status=doc.status,
            created_at=doc.created_at,
            updated_at=doc.updated_at,
        )
