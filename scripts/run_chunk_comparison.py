"""
Chunk 策略对比 + Embedding/FAISS 全链路验证脚本

用法:
    python scripts/run_chunk_comparison.py

产物:
    docs/week9/chunk_comparison.md（3 种策略对比表）
    LangFuse Dashboard 追踪记录（chunk → embed → search 全链路）
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.parsing.chunking import ChunkComparator, get_chunker  # noqa: E402
from app.embedding import DashscopeEmbeddingClient  # noqa: E402
from app.vector import FaissIndex  # noqa: E402
from app.config import settings  # noqa: E402


def load_docs(doc_dir: Path) -> list[str]:
    """读取测试文档（txt/md 格式，递归子目录）"""
    texts = []
    for ext in ("*.txt", "*.md"):
        for f in sorted(doc_dir.rglob(ext)):
            try:
                text = f.read_text(encoding="utf-8")
                if text.strip():  # 跳过空文件
                    texts.append(text)
            except (UnicodeDecodeError, OSError):
                continue
    return texts


async def verify_pipeline(sample_text: str) -> None:
    """
    全链路验证：切分 → embedding → FAISS 索引 → 检索。
    这是第 10 周 RAG 闭环的前置链路。
    """
    print("\n" + "=" * 60)
    print("全链路验证：sentence 切分 → embedding → FAISS → 检索")
    print("=" * 60)

    # 1. 切分
    chunker = get_chunker("sentence")
    chunks = chunker.chunk(sample_text)
    print(f"[1] 切分: {len(chunks)} 个 chunk")

    # 2. embedding
    embedder = DashscopeEmbeddingClient()
    texts = [c.text for c in chunks[:20]]  # 只取前 20 个，避免超量
    vectors = await embedder.embed(texts)
    print(f"[2] embedding: {len(vectors)} 个向量（维度 {len(vectors[0])}）")

    # 3. FAISS 索引
    index = FaissIndex(dim=settings.EMBEDDING_DIM)
    index.add(vectors)
    print(f"[3] FAISS: 索引 {len(index)} 条")

    # 4. 检索验证
    query_vec = await embedder.embed_one("什么是 RAG？")
    results = index.search(query_vec, top_k=3)
    print(f"[4] 检索 top 3:")
    for chunk_id, score in results:
        preview = texts[chunk_id][:40].replace("\n", " ")
        print(f"    id={chunk_id} score={score:.4f} text='{preview}...'")


async def main() -> None:
    doc_dir = Path(__file__).resolve().parents[1] / "test_docs"
    texts = load_docs(doc_dir)

    if not texts:
        print("未找到测试文档（test_docs 下无 txt/md 文件），跳过对比")
        return

    # ── Chunk 策略对比 ──
    print("=" * 60)
    print(f"Chunk 策略对比（{len(texts)} 篇文档）")
    print("=" * 60)

    comparator = ChunkComparator()
    report = comparator.compare(texts)
    print(report.comparison_table)

    # 写对比表
    out_dir = Path(__file__).resolve().parents[1] / "docs" / "week9"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chunk_comparison.md"
    out_path.write_text(report.comparison_table, encoding="utf-8")
    print(f"\n对比表已写入: {out_path}")

    # ── 全链路验证 ──
    sample = max(texts, key=len)  # 取最长文档做验证
    await verify_pipeline(sample)


if __name__ == "__main__":
    asyncio.run(main())
