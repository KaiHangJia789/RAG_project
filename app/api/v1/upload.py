"""
文件上传接口
POST /api/v1/upload — 上传文档文件 + 自动解析入库
"""
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File

from app.config import settings
from app.dependencies.auth import DocumentServiceDep, ParsingServiceDep
from app.exceptions.handlers import FileValidationError
from app.models.document import DocumentResponse, DocumentStatus
from app.models.response import APIResponse

logger = logging.getLogger("rag_api.upload")

router = APIRouter(prefix="/upload", tags=["文件上传"])

# 需要解析的文件类型
PARSEABLE_TYPES = {".pdf", ".md", ".txt"}


async def _index_uploaded(document_id: str, filename: str) -> None:
    """
    把刚落库的 chunk 写入向量索引，使新文档立即可检索。

    失败只告警不抛异常 —— 上传本身已成功，索引可随时用
    scripts/build_index.py 重建，不该因为索引问题让用户看到上传失败。
    """
    from app.db.repositories.chunk_repo import ChunkRepository
    from app.dependencies import auth
    from app.services.index_service import IndexRecord

    svc = auth._index_service
    db = auth._db
    if svc is None or db is None:
        return

    try:
        repo = ChunkRepository()
        async with db.pool.acquire() as conn:
            rows = await repo.fetch_by_document(conn, document_id, svc.strategy)

        records = [
            IndexRecord(
                faiss_id=r["faiss_id"],
                chunk_id=str(r["id"]),
                document_id=str(r["document_id"]),
                filename=filename,
                chunk_index=r["chunk_index"],
                text=r["chunk_text"],
                page_number=r["page_number"],
                chunk_strategy=r["chunk_strategy"],
            )
            for r in rows
            if r["faiss_id"] is not None
        ]
        if not records:
            return

        added = await svc.add_records(records)
        logger.info("文档已入索引: %s → %d 条向量", filename, added)

    except Exception as e:
        logger.error(
            "入索引失败（文档已上传成功，可用 scripts/build_index.py 重建）: %s — %s",
            filename, e,
        )


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
    final_status = DocumentStatus.READY

    if ext in PARSEABLE_TYPES:
        logger.info("开始解析: %s (doc_id=%s)", file.filename, doc.id)
        try:
            # 用服务当前配置的策略切分 —— 标签与切分方式一致，
            # 检索时才能在当前策略的索引里命中（否则上传了也检索不到）
            parsed, chunk_ids = await parsing_service.parse_and_persist(
                filename=file.filename,
                content=content,
                document_id=doc.id,
                chunk_strategy=settings.CHUNK_STRATEGY,
            )
            logger.info(
                "解析完成: %s → %d blocks, %d chunks",
                file.filename, parsed.total_blocks, len(chunk_ids),
            )
            if parsed.warnings:
                for w in parsed.warnings:
                    logger.warning("  %s", w)
        except Exception as e:
            logger.error("解析失败: %s (doc_id=%s): %s", file.filename, doc.id, e)
            final_status = DocumentStatus.FAILED

    # 5. 状态落库（必须写回 DB，否则列表/详情页读到的永远是 uploaded）
    if final_status != doc.status:
        try:
            await doc_service.update_status(doc.id, final_status)
            doc.status = final_status
        except Exception as e:
            # 状态写回失败不该让整个上传请求失败——文件和 DB 记录都已就绪
            logger.error(
                "状态写回失败: doc_id=%s 目标状态=%s: %s（文档已上传，DB 中仍为 %s）",
                doc.id, final_status.value, e, doc.status.value,
            )

    # 6. 写入向量索引（Week10）—— 让新上传的文档立即可检索
    #    失败只告警不阻断：文档已上传成功，索引可由 scripts/build_index.py 重建
    if final_status == DocumentStatus.READY:
        await _index_uploaded(doc.id, file.filename)

    return APIResponse(
        code=201,
        message="文档上传成功" if doc.status != DocumentStatus.FAILED else "文档已上传，但解析失败",
        data=doc,
    )
