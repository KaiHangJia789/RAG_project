"""
状态序列化辅助

**为什么需要这一层**：
  LangGraph 的 State 必须是可序列化的（SQLite checkpointer 要落盘）。
  但 `IndexHit` / `Citation` 是 Pydantic 模型，直接放进 State 会有两个问题：
    1. checkpointer 序列化时依赖具体类型，跨版本易碎
    2. 调试时看到的是一堆对象 repr，不直观

  所以约定：**State 里只放 dict**，进出节点时用本模块转换。

  注意 `IndexHit` 有便捷 property（.text/.filename 等），转 dict 时要把
  record 一起展开，否则下游拿不到这些字段。
"""
from app.rag.models import Citation
from app.services.index_service import IndexHit


def hit_to_dict(hit: IndexHit) -> dict:
    """
    IndexHit → 可序列化 dict。

    把 record 摊平到顶层（而不是嵌套保留），让下游节点用 `h["text"]`
    直接取值，与 `IndexHit.text` 这个 property 的用法保持一致。
    """
    r = hit.record
    return {
        "faiss_id": hit.faiss_id,
        "score": hit.score,
        "chunk_id": r.chunk_id,
        "document_id": r.document_id,
        "filename": r.filename,
        "chunk_index": r.chunk_index,
        "page_number": r.page_number,
        "chunk_strategy": r.chunk_strategy,
        "text": r.text,
    }


def dict_to_hit(d: dict) -> IndexHit:
    """dict → IndexHit（需要重新包装成 IndexRecord）"""
    from app.services.index_service import IndexRecord

    return IndexHit(
        faiss_id=d["faiss_id"],
        score=d["score"],
        record=IndexRecord(
            faiss_id=d["faiss_id"],
            chunk_id=d["chunk_id"],
            document_id=d["document_id"],
            filename=d["filename"],
            chunk_index=d["chunk_index"],
            text=d["text"],
            page_number=d.get("page_number"),
            chunk_strategy=d.get("chunk_strategy", "splitter"),
        ),
    )


def citation_to_dict(c: Citation) -> dict:
    """Citation → dict（Pydantic 的 model_dump 已经够用，这里统一入口）"""
    return c.model_dump()


def dict_to_citation(d: dict) -> Citation:
    return Citation(**d)


def hits_to_dicts(hits: list[IndexHit]) -> list[dict]:
    return [hit_to_dict(h) for h in hits]


def dicts_to_hits(dicts: list[dict]) -> list[IndexHit]:
    return [dict_to_hit(d) for d in dicts]


def answer_text_of(hits_as_dicts: list[dict]) -> str:
    """把检索结果拼成纯文本（供需要文本形式的下游用，例如判官评分）"""
    return "\n\n".join(h.get("text", "") for h in hits_as_dicts)
