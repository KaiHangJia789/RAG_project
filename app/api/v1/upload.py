"""
文件上传接口
POST /api/v1/upload — 上传文档文件
"""
from fastapi import APIRouter, UploadFile, File

from app.dependencies.auth import DocumentServiceDep
from app.exceptions.handlers import FileValidationError
from app.models.document import DocumentResponse
from app.models.response import APIResponse

router = APIRouter(prefix="/upload", tags=["文件上传"])


@router.post(
    "",
    summary="上传文档",
    description="""
上传一个文档文件到服务器。

**支持的格式**: PDF, TXT, MD, DOCX, CSV, JSON, XML
**大小限制**: 50MB

上传成功后返回文档的元信息，包括文档ID、文件名、类型、大小等。
后续可调用 `GET /api/v1/documents/{id}` 获取文档详情。
    """,
    response_model=APIResponse[DocumentResponse],
    status_code=201,
)
async def upload_document(
    file: UploadFile = File(
        ...,
        title="文档文件",
        description="要上传的文档文件",
    ),
    doc_service: DocumentServiceDep = None,
):
    """
    上传单个文档文件。

    - **file**: 通过 multipart/form-data 上传的文件
    - 文件会经过类型和大小校验
    - 校验通过后存储到服务器本地磁盘
    - 返回包含文档ID的元数据，客户端可用此ID进行后续操作
    """
    # 1. 读取文件内容
    content = await file.read()

    # 2. 校验文件名和大小
    if not file.filename:
        raise FileValidationError("文件名为空")

    # 3. 调用业务层创建文档
    try:
        doc = await doc_service.create_document(
            filename=file.filename,
            content=content,
        )
    except ValueError as e:
        raise FileValidationError(str(e))

    return APIResponse(
        code=201,
        message="文档上传成功",
        data=doc,
    )
