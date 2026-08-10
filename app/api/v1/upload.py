"""
文件上传接口
POST /api/v1/upload — 上传文档文件 + 自动解析入库
"""
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File

from app.dependencies.auth import DocumentServiceDep, ParsingServiceDep
from app.exceptions.handlers import FileValidationError
from app.models.document import DocumentResponse, DocumentStatus
from app.models.response import APIResponse

logger = logging.getLogger("rag_api.upload")

router = APIRouter(prefix="/upload", tags=["文件上传"])

# 需要解析的文件类型
PARSEABLE_TYPES = {".pdf", ".md", ".txt"}


@router.post(
    "",
    summary="上传文档（自动解析）",
    description="""
上传一个文档文件到服务器，自动执行内容解析。

**支持的格式**: PDF, TXT, MD, DOCX, CSV, JSON, XML
**自动解析**: PDF/MD/TXT 格式上传后自动提取文本内容并写入 chunks 表
**大小限制**: 50MB

上传成功后返回文档的元信息。解析过程异步进行，状态字段表示处理进度：
- uploaded → 上传完成
- parsing → 正在解析
- ready → 解析完成
- failed → 解析失败
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
    parsing_service: ParsingServiceDep = None,
):
    # 1. 读取文件内容
    content = await file.read()

    # 2. 校验文件名
    if not file.filename:
        raise FileValidationError("文件名为空")

    # 3. 写入文件 + 数据库
    try:
        doc = await doc_service.create_document(
            filename=file.filename,
            content=content,
        )
    except ValueError as e:
        raise FileValidationError(str(e))

    # 4. 自动解析（PDF/MD/TXT 格式）
    ext = Path(file.filename).suffix.lower()
    if ext in PARSEABLE_TYPES:
        logger.info("开始解析: %s (doc_id=%s)", file.filename, doc.id)
        try:
            parsed, chunk_ids = await parsing_service.parse_and_persist(
                filename=file.filename,
                content=content,
                document_id=doc.id,
            )
            logger.info(
                "解析完成: %s → %d blocks, %d chunks",
                file.filename, parsed.total_blocks, len(chunk_ids),
            )
            if parsed.warnings:
                for w in parsed.warnings:
                    logger.warning("  %s", w)

            # 更新状态为 ready
            doc.status = DocumentStatus.READY
        except Exception as e:
            logger.error("解析失败: %s (doc_id=%s): %s", file.filename, doc.id, e)
            doc.status = DocumentStatus.FAILED
    else:
        # 非可解析格式（如 CSV/JSON）— 标记为 ready 不需要解析
        doc.status = DocumentStatus.READY

    return APIResponse(
        code=201,
        message="文档上传成功" if doc.status != DocumentStatus.FAILED else "文档已上传，但解析失败",
        data=doc,
    )
