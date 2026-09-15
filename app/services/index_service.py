"""
IndexService — FAISS 向量索引的生命周期与检索

设计要点:
  1. **每种 chunk 策略一个独立索引目录**：data/index/{strategy}/{index.faiss, meta.json}
     对比实验靠这个隔离，互不干扰。
  2. **元数据走 sidecar JSON，不查库**。检索命中后需要 chunk 文本/文件名/页码，
     这些和向量一起落盘成 meta.json。好处：
       - 检索路径变成纯内存查找 → 单元测试可完整覆盖（DB 测试替身对复杂
         WHERE/JOIN 支持有限，会静默返回全表，在线路径依赖 SQL 就等于放弃可测性）
       - 少一次数据库往返
  3. **写入必须串行化**：FAISS 索引是进程内可变状态，并发 add + save 会互相覆盖。
  4. **单 worker 约束**：索引在进程内存里，多 worker 各持一份、互相覆盖落盘文件。
     本方案只支持 `--workers 1`；要横向扩展得换 pgvector。
"""
import asyncio
import json
import logging
from pathlib import Path

from pydantic import BaseModel

from app.config import settings
from app.embedding.base import EmbeddingClient
from app.vector import FaissIndex

logger = logging.getLogger("rag_api.index")


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

class IndexRecord(BaseModel):
    """sidecar 里的一条记录 —— 检索命中后返回给上层的全部信息"""

    faiss_id: int
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    text: str
    page_number: int | None = None
    chunk_strategy: str = "splitter"


class IndexHit(BaseModel):
    """一次检索命中"""

    faiss_id: int
    score: float
    record: IndexRecord

    # ── 便捷透传（前端/引用来源直接用）──

    @property
    def text(self) -> str:
        return self.record.text

    @property
    def filename(self) -> str:
        return self.record.filename

    @property
    def chunk_id(self) -> str:
        return self.record.chunk_id

    @property
    def page_number(self) -> int | None:
        return self.record.page_number


# ═══════════════════════════════════════════════════════════════
# 服务
# ═══════════════════════════════════════════════════════════════

class IndexService:
    """向量索引服务（加载/写入/检索/持久化）"""

    def __init__(
        self,
        embedder: EmbeddingClient | None = None,
        *,
        strategy: str | None = None,
        index_dir: Path | None = None,
        dim: int | None = None,
    ) -> None:
        self.strategy = strategy or settings.CHUNK_STRATEGY
        self.dim = dim or settings.EMBEDDING_DIM
        self._index_dir = Path(index_dir) if index_dir else settings.INDEX_DIR
        self._embedder = embedder
        self._index = FaissIndex(self.dim)
        self._records: dict[int, IndexRecord] = {}
        self._lock = asyncio.Lock()
        self._loaded = False

    # ═══════════════════════════════════════════════════════════
    # 路径与状态
    # ═══════════════════════════════════════════════════════════

    @property
    def index_dir(self) -> Path:
        return self._index_dir / self.strategy

    @property
    def index_path(self) -> Path:
        return self.index_dir / "index.faiss"

    @property
    def meta_path(self) -> Path:
        return self.index_dir / "meta.json"

    @property
    def size(self) -> int:
        return self._index.size

    @property
    def ready(self) -> bool:
        """索引是否可用（有条目即认为可用）"""
        return self._index.size > 0

    # ═══════════════════════════════════════════════════════════
    # 加载 / 保存
    # ═══════════════════════════════════════════════════════════

    def load(self) -> None:
        """
        从磁盘加载索引与元数据。

        文件不存在 → 保持空索引并告警（不抛异常，让应用能启动；
        检索时会因为 ready=False 而拒答，并给出可操作的提示）。
        """
        if self._loaded:
            return

        if not self.index_path.exists() or not self.meta_path.exists():
            logger.warning(
                "索引不存在（策略=%s，路径=%s），检索将不可用。"
                "请先运行: python scripts/build_index.py --strategy %s",
                self.strategy, self.index_dir, self.strategy,
            )
            self._loaded = True
            return

        self._index = FaissIndex.load(self.index_path, self.dim)
        raw = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self._records = {
            int(item["faiss_id"]): IndexRecord(**item) for item in raw["records"]
        }
        self._loaded = True

        logger.info(
            "索引已加载: 策略=%s, 向量=%d 条, 元数据=%d 条",
            self.strategy, self._index.size, len(self._records),
        )

    def save(self) -> None:
        """落盘（索引 + 元数据）"""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._index.save(self.index_path)

        payload = {
            "strategy": self.strategy,
            "dim": self.dim,
            "count": len(self._records),
            "records": [r.model_dump() for r in self._records.values()],
        }
        # 先写临时文件再替换 —— 避免写一半崩溃留下损坏的 meta.json
        tmp = self.meta_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(self.meta_path)

    # ═══════════════════════════════════════════════════════════
    # 写入
    # ═══════════════════════════════════════════════════════════

    async def add_records(self, records: list[IndexRecord]) -> int:
        """
        向量化并追加一批记录，返回新增条数。

        已存在的 faiss_id 会被跳过 —— FAISS 的 add_with_ids 对重复 id **不报错**，
        只会追加一条新记录，导致两条不同向量共享同一 id、top_k 名额被挤占。
        这个防护不能省。
        """
        if not records:
            return 0
        if self._embedder is None:
            raise RuntimeError("IndexService 未注入 embedder，无法向量化")

        pending = [r for r in records if not self._index.has_id(r.faiss_id)]
        skipped = len(records) - len(pending)
        if skipped:
            logger.warning("跳过 %d 条已存在的向量（faiss_id 重复）", skipped)
        if not pending:
            return 0

        # embedding 是网络调用，放在锁外 await —— 否则并发上传会被串行化
        vectors = await self._embedder.embed([r.text for r in pending])
        # 按 faiss_id 建映射，锁内过滤时才不会出现「向量与记录错位」
        vec_by_id = {r.faiss_id: v for r, v in zip(pending, vectors)}

        async with self._lock:
            # 双重检查：等锁期间可能已被另一个协程写入
            fresh = [r for r in pending if not self._index.has_id(r.faiss_id)]
            if not fresh:
                return 0
            self._index.add(
                [vec_by_id[r.faiss_id] for r in fresh],
                ids=[r.faiss_id for r in fresh],
            )
            for r in fresh:
                self._records[r.faiss_id] = r
            self.save()

        logger.info("索引新增 %d 条（策略=%s，当前 %d 条）",
                    len(fresh), self.strategy, self._index.size)
        return len(fresh)

    def rebuild(
        self, records: list[IndexRecord], vectors: list[list[float]]
    ) -> int:
        """
        全量重建索引（幂等）。**这是脚本侧唯一的写入路径。**

        重复 ingest 时用 add 会留下重复 id 条目，只有全量重建能彻底清理。
        """
        if len(records) != len(vectors):
            raise ValueError(
                f"记录数({len(records)})与向量数({len(vectors)})不一致"
            )

        self._index = FaissIndex(self.dim)
        self._records = {}
        if records:
            self._index.add(vectors, ids=[r.faiss_id for r in records])
            self._records = {r.faiss_id: r for r in records}
        self.save()

        logger.info("索引已重建: 策略=%s, %d 条", self.strategy, len(self._records))
        return len(records)

    def remove_document(self, document_id: str) -> int:
        """
        删除某文档的全部向量与元数据，返回删除条数。

        不删的话索引里会留下指向已删 chunk 的悬空向量 —— 检索命中后
        回表取不到文本，引用来源就成了空指针。
        """
        victims = [
            fid for fid, r in self._records.items() if r.document_id == document_id
        ]
        if not victims:
            return 0

        removed = self._index.remove_ids(victims)
        for fid in victims:
            self._records.pop(fid, None)
        self.save()

        logger.info("已从索引移除文档 %s 的 %d 条向量", document_id, removed)
        return removed

    # ═══════════════════════════════════════════════════════════
    # 检索
    # ═══════════════════════════════════════════════════════════

    async def search(
        self,
        query: str,
        *,
        retrieve_k: int | None = None,
        min_score: float | None = None,
        final_k: int | None = None,
        document_id: str | None = None,
    ) -> list[IndexHit]:
        """
        语义检索。

        先取 retrieve_k 条再按 min_score 过滤、截断到 final_k。这样一次 FAISS 调用
        可以服务多组 (阈值, final_k) 配置 —— 参数扫描时同一问题只需 embed 一次，
        而 embedding 调用是扫描成本的大头。

        Args:
            retrieve_k: FAISS 单次取回条数（默认 settings.RAG_RETRIEVE_K）
            min_score: 相似度阈值，低于此值丢弃
            final_k: 最终返回条数
            document_id: 限定在某文档内检索（None = 全库）
        """
        if not self._index.size:
            return []
        if self._embedder is None:
            raise RuntimeError("IndexService 未注入 embedder，无法检索")

        retrieve_k = retrieve_k if retrieve_k is not None else settings.RAG_RETRIEVE_K
        min_score = min_score if min_score is not None else settings.RAG_MIN_SCORE
        final_k = final_k if final_k is not None else settings.RAG_FINAL_K

        query_vec = await self._embedder.embed_one(query)
        raw = self._index.search(query_vec, top_k=retrieve_k)

        hits: list[IndexHit] = []
        for faiss_id, score in raw:
            if document_id is not None:
                record = self._records.get(faiss_id)
                if record is None or record.document_id != document_id:
                    continue
            else:
                record = self._records.get(faiss_id)
            if record is None:
                # 索引有向量但 sidecar 无记录 → 数据不一致，跳过并告警
                logger.warning("faiss_id=%d 在元数据中缺失，跳过", faiss_id)
                continue
            if score < min_score:
                continue
            hits.append(IndexHit(faiss_id=faiss_id, score=score, record=record))
            if len(hits) >= final_k:
                break

        return hits

    # ═══════════════════════════════════════════════════════════
    # 查询辅助
    # ═══════════════════════════════════════════════════════════

    def record_of(self, faiss_id: int) -> IndexRecord | None:
        return self._records.get(faiss_id)

    def get_by_chunk_id(self, chunk_id: str) -> IndexRecord | None:
        for r in self._records.values():
            if r.chunk_id == chunk_id:
                return r
        return None

    def all_records(self) -> list[IndexRecord]:
        return list(self._records.values())

    def stats(self) -> dict:
        """索引概况（供 /rag/stats 与自检脚本用）"""
        pages = {r.page_number for r in self._records.values() if r.page_number}
        return {
            "strategy": self.strategy,
            "dim": self.dim,
            "vectors": self._index.size,
            "records": len(self._records),
            "documents": len({r.document_id for r in self._records.values()}),
            "pages": len(pages),
            "index_path": str(self.index_path),
            "exists": self.index_path.exists(),
        }
