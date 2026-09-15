"""
问答接口
POST /api/v1/qa/ask            — 检索增强问答（带引用来源）
GET  /api/v1/qa/chunks/{id}    — 引用定位：取 chunk 及前后文
GET  /api/v1/qa/stats          — 索引概况
"""
import logging

from fastapi import APIRouter, Query

from app.dependencies.auth import DbDep, RagPipelineDep
from app.exceptions.handlers import ChunkNotFoundError, IndexNotReadyError
from app.models.qa import ChunkContext, QARequest, QAStats
from app.models.response import APIResponse
from app.rag.models import RetrievalConfig

logger = logging.getLogger("rag_api.qa")

router = APIRouter(prefix="/qa", tags=["问答"])


@router.post(
    "/ask",
    summary="检索增强问答",
    description="""
基于已索引的文档回答问题，返回答案与引用来源。

**流程**: 向量检索 → 相似度阈值过滤 → 装配带编号上下文 → LLM 生成 → 解析引用

**拒答**: 以下情况会拒答而不调用 LLM（`refused=true`）
- 索引未构建
- 检索结果全部低于 `min_score` 阈值
- LLM 判断上下文不足以回答

**引用**: `citations[].index` 对应答案中的 `[n]` 标记，
`chunk_id` 可用于 `GET /api/v1/qa/chunks/{chunk_id}` 定位原文。
    """,
    response_model=APIResponse[dict],
)
async def ask_question(
    body: QARequest,
    pipeline: RagPipelineDep = None,
):
    config = RetrievalConfig(
        strategy=body.strategy,
        retrieve_k=body.retrieve_k,
        min_score=body.min_score,
        final_k=body.final_k,
        prompt_name=body.prompt_name,
        document_id=body.document_id,
        temperature=body.temperature,
    )

    result = await pipeline.answer(body.question, config)

    return APIResponse(
        code=200,
        message="已回答" if not result.refused else "已拒答",
        data=result.model_dump(mode="json"),
    )


@router.get(
    "/chunks/{chunk_id}",
    summary="获取 chunk 及上下文",
    description="""
按 chunk_id 取回该文本块及其前后相邻块 —— 这是「引用来源可定位」的落点。

返回结果中 `is_target=true` 的那条即被引用的原文块，
前端应滚动定位到它并高亮。
    """,
    response_model=APIResponse[ChunkContext],
)
async def get_chunk_context(
    chunk_id: str,
    before: int = Query(default=1, ge=0, le=10, description="向前多取几块"),
    after: int = Query(default=1, ge=0, le=10, description="向后多取几块"),
    db: DbDep = None,
    pipeline: RagPipelineDep = None,
):
    from app.db.repositories.chunk_repo import ChunkRepository

    repo = ChunkRepository()

    # 先按主键取目标块（拿 document_id 与 chunk_index）
    async with db.pool.acquire() as conn:
        target = await repo.fetch_one_by_id(conn, chunk_id)

    if target is None:
        # 给一条能定位问题的提示：区分"索引里有但库里没有"与"完全不存在"
        record = (
            pipeline.index_service.get_by_chunk_id(chunk_id)
            if pipeline.index_service
            else None
        )
        hint = None
        if record is not None:
            hint = (
                f"该块存在于向量索引（文档 '{record.filename}' 第 {record.chunk_index} 块），"
                f"但数据库中查不到，索引与数据库不一致，请重建索引"
            )
        raise ChunkNotFoundError(chunk_id, hint=hint)

    row = dict(target)

    async with db.pool.acquire() as conn:
        rows = await repo.fetch_neighbors(
            conn, str(row["document_id"]), int(row["chunk_index"]),
            before=before, after=after,
        )

    context = ChunkContext(
        document_id=str(row["document_id"]),
        filename=row.get("filename") or "",
        chunk_strategy=row.get("chunk_strategy", "splitter"),
        items=[
            {
                "chunk_id": str(r["id"]),
                "chunk_index": r["chunk_index"],
                "text": r["chunk_text"],
                "page_number": r["page_number"],
                "is_target": str(r["id"]) == chunk_id,
            }
            for r in rows
        ],
    )
    return APIResponse(code=200, message="查询成功", data=context)


@router.get(
    "/stats",
    summary="索引概况",
    description="返回当前向量索引的策略、向量数、来源文档数等信息",
    response_model=APIResponse[QAStats],
)
async def get_stats(pipeline: RagPipelineDep = None):
    svc = pipeline.index_service
    if svc is None:
        raise IndexNotReadyError()

    stats = svc.stats()
    return APIResponse(
        code=200,
        message=f"索引 {stats['vectors']} 条向量",
        data=QAStats(**stats),
    )
