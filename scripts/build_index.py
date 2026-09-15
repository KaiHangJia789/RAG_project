"""
构建向量索引 — 两步走：入库（解析切分写 chunks 表） + 建索引（从 chunks 表向量化）

用法:
    python scripts/build_index.py                          # 默认策略
    python scripts/build_index.py --all                    # 全部策略（对比实验用）
    python scripts/build_index.py --verify                 # 构建后做一致性自检

核心设计（修复了「上传文档检索不到」的 bug）:

  **索引重建从 chunks 表读 chunk_text 向量化，不重新解析物理文件。**

  之前的实现重新解析物理文件，有两个致命缺陷：
    1. 只遍历 data/corpus/ 语料目录，漏掉用户上传的文档（在 uploads/ 里），
       rebuild 全量重建时把它们的向量清掉了
    2. 依赖物理文件存在 —— 文件一旦丢失就无法重建

  改成从 chunks 表读之后：
    - 物理文件丢失不影响重建（chunk_text 已经持久化在库里）
    - 用户上传的文档（chunks 表里已有）自动被索引
    - 索引与数据库天然一致（索引直接由 chunks 表驱动）

  「入库」（ingest_corpus）才需要物理文件，且只处理 data/corpus/ 语料目录；
  用户上传的文档由 upload 流程入库，不经过这里。
"""
import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.db.connection import Database  # noqa: E402
from app.db.repositories.chunk_repo import ChunkRepository  # noqa: E402
from app.embedding.dashscope_embedding import DashscopeEmbeddingClient  # noqa: E402
from app.parsing.parser_registry import get_default_registry  # noqa: E402
from app.services.chunking_service import ChunkingService, list_presets  # noqa: E402
from app.services.index_service import IndexRecord, IndexService  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_index")

# 语料归属的演示用户（与 config.DEFAULT_USERNAME 一致）
CORPUS_USERNAME = settings.DEFAULT_USERNAME


# ═══════════════════════════════════════════════════════════════
# 语料登记与入库
# ═══════════════════════════════════════════════════════════════

def load_corpus(corpus_dir: Path) -> list[Path]:
    """加载语料目录里的 .md/.txt 文件（用于入库）"""
    if not corpus_dir.exists():
        logger.info("语料目录不存在（%s），跳过语料入库", corpus_dir)
        return []

    files = sorted(
        [p for p in corpus_dir.rglob("*.md")] + [p for p in corpus_dir.rglob("*.txt")]
    )
    valid: list[Path] = []
    skipped: list[tuple[Path, str]] = []
    for f in files:
        try:
            if not f.read_text(encoding="utf-8").strip():
                skipped.append((f, "内容为空"))
                continue
            valid.append(f)
        except (UnicodeDecodeError, OSError) as e:
            skipped.append((f, str(e)))

    if skipped:
        logger.warning("跳过 %d 个语料文件：", len(skipped))
        for f, reason in skipped:
            logger.warning("   %s — %s", f.name, reason)

    logger.info("语料目录 %d 个文件", len(valid))
    return valid


async def ensure_document(db: Database, user_id: str, corpus_file: Path, text: str) -> str:
    """确保语料文档在 documents 表里存在（幂等），返回 document_id"""
    filename = corpus_file.name
    row = await db.fetch_one(
        "SELECT id FROM documents WHERE user_id = $1 AND filename = $2",
        user_id, filename,
    )
    if row:
        return str(row["id"])

    now = datetime.now(UTC)
    row = await db.fetch_one(
        """
        INSERT INTO documents
            (user_id, filename, file_type, file_size, storage_path, status,
             created_at, updated_at)
        VALUES ($1,$2,$3,$4,$5,'ready',$6,$6)
        RETURNING id
        """,
        user_id, filename, corpus_file.suffix.lower(),
        len(text.encode("utf-8")), f"data/corpus/{filename}", now,
    )
    return str(row["id"])


async def ingest_corpus(db: Database, corpus_dir: Path, strategies: list[str]) -> None:
    """
    把语料文件解析切分（所有策略），写入 chunks 表。

    幂等：persist_chunks 先删同 (文档, 策略) 的旧块再插入。
    """
    import hashlib

    user_id = await _get_user_id(db)
    parser_registry = get_default_registry()

    for path in load_corpus(corpus_dir):
        text = path.read_text(encoding="utf-8")
        doc_id = await ensure_document(db, user_id, path, text)
        parser = parser_registry.get(path.suffix.lower())
        parsed = await parser.parse(path.name, path.read_bytes())

        for strategy in strategies:
            svc = ChunkingService(strategy)
            chunks = svc.chunk_blocks(parsed.blocks)

            async with db.transaction() as conn:
                await conn.execute(
                    'DELETE FROM "chunks" WHERE document_id = $1 AND chunk_strategy = $2',
                    doc_id, strategy,
                )
                for i, c in enumerate(chunks):
                    clean = c.text.strip()
                    if not clean:
                        continue
                    await conn.fetchrow(
                        """
                        INSERT INTO "chunks"
                            (document_id, chunk_strategy, chunk_index, chunk_text,
                             chunk_hash, token_count, page_number)
                        VALUES ($1,$2,$3,$4,$5,$6,$7)
                        RETURNING id, faiss_id
                        """,
                        doc_id, strategy, i, clean,
                        hashlib.sha256(clean.encode()).hexdigest(),
                        max(1, int(len(clean) * 0.5)),
                        c.page_number,
                    )

            m = svc.quality_metrics(chunks)
            logger.info(
                "   %-28s %-12s %2d 块 均长%4d",
                path.name, strategy, m["count"], m["avg_len"],
            )


# ═══════════════════════════════════════════════════════════════
# 建索引（从 chunks 表，不依赖物理文件）
# ═══════════════════════════════════════════════════════════════

async def rebuild_index_from_db(db: Database, strategy: str) -> int:
    """
    从 chunks 表读该策略的所有 chunk，向量化，重建 FAISS 索引。

    这是索引重建的**唯一正确来源** —— chunk_text 已经持久化在库里，
    物理文件丢失不影响重建，用户上传的文档也会被索引。
    """
    repo = ChunkRepository()
    async with db.pool.acquire() as conn:
        rows = await repo.fetch_for_indexing(conn, strategy)

    records = [
        IndexRecord(
            faiss_id=int(r["faiss_id"]),
            chunk_id=str(r["id"]),
            document_id=str(r["document_id"]),
            filename=r["filename"],
            chunk_index=r["chunk_index"],
            text=r["chunk_text"],
            page_number=r["page_number"],
            chunk_strategy=r["chunk_strategy"],
        )
        for r in rows
    ]

    if not records:
        logger.warning("策略 '%s' 在 chunks 表中无数据，跳过", strategy)
        return 0

    logger.info("策略 '%s'：向量化 %d 条 chunk ...", strategy, len(records))
    embedder = DashscopeEmbeddingClient()
    vectors = await embedder.embed([r.text for r in records])

    index = IndexService(embedder, strategy=strategy)
    index.rebuild(records, vectors)

    logger.info("策略 '%s' 完成：%d 条向量", strategy, index.size)
    return index.size


async def _get_user_id(db: Database) -> str:
    """按用户名取 user_id（不硬编码 UUID，换库后依然有效）"""
    row = await db.fetch_one(
        "SELECT id FROM users WHERE username = $1", CORPUS_USERNAME
    )
    if row is None:
        raise RuntimeError(
            f"用户 '{CORPUS_USERNAME}' 不存在，请先执行 "
            f"psql -f app/db/migrations/seed.sql"
        )
    return str(row["id"])


# ═══════════════════════════════════════════════════════════════
# 一致性自检
# ═══════════════════════════════════════════════════════════════

async def verify(db: Database, strategies: list[str]) -> bool:
    """一致性自检：索引向量数 == DB 中已分配 faiss_id 的 chunk 数"""
    repo = ChunkRepository()
    ok = True

    for strategy in strategies:
        async with db.pool.acquire() as conn:
            db_count = await repo.count_indexed(conn, strategy)

        svc = IndexService(strategy=strategy)
        svc.load()
        idx_count = svc.size

        matched = db_count == idx_count
        ok = ok and matched
        logger.info(
            "   %-12s DB=%d 索引=%d %s",
            strategy, db_count, idx_count, "✓" if matched else "✗ 不一致",
        )

    return ok


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

async def main() -> None:
    parser = argparse.ArgumentParser(description="构建向量索引")
    parser.add_argument("--strategy", default=settings.CHUNK_STRATEGY,
                        choices=list_presets(), help="切分策略")
    parser.add_argument("--all", action="store_true", help="构建全部策略")
    parser.add_argument("--corpus", type=Path, default=settings.CORPUS_DIR)
    parser.add_argument("--verify", action="store_true", help="构建后自检")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="跳过语料入库，只从 chunks 表重建索引")
    args = parser.parse_args()

    strategies = list_presets() if args.all else [args.strategy]

    db = Database()
    await db.connect(settings.DATABASE_DSN)
    try:
        # 1. 入库：语料文件解析切分写 chunks 表（需要物理文件）
        if not args.skip_ingest:
            logger.info("=" * 60)
            logger.info("入库语料（策略: %s）", ", ".join(strategies))
            await ingest_corpus(db, args.corpus, strategies)

        # 2. 建索引：从 chunks 表向量化（不依赖物理文件）
        logger.info("=" * 60)
        logger.info("从 chunks 表重建索引")
        for strategy in strategies:
            await rebuild_index_from_db(db, strategy)

        # 3. 一致性自检
        if args.verify:
            logger.info("=" * 60)
            logger.info("一致性自检：")
            ok = await verify(db, strategies)
            if not ok:
                logger.error("自检未通过 —— 索引与数据库不一致，请重跑构建")
                sys.exit(1)
            logger.info("自检通过 ✓")
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
