"""
IndexService 测试 — 落盘/加载往返、重复 id 防护、删除文档

用确定性假嵌入（按文本哈希生成）避免网络依赖，同时保证同文本同向量。
"""
import asyncio
import pathlib

import pytest

from app.services.index_service import IndexHit, IndexRecord, IndexService


class FakeEmbedder:
    """确定性假嵌入：同文本必得同向量，不同文本方向不同"""

    def __init__(self, dim: int = 8):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for i, ch in enumerate(text[: self.dim]):
            v[i] = (ord(ch) % 50) / 50.0
        # 用哈希把幅度集中到某一维，保证不同文本可区分
        v[abs(hash(text)) % self.dim] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    async def embed(self, texts):
        return [self._vec(t) for t in texts]

    async def embed_one(self, text):
        return self._vec(text)


def make_records(n: int, doc_id: str = "doc-1") -> list[IndexRecord]:
    return [
        IndexRecord(
            faiss_id=i, chunk_id=f"chunk-{i}", document_id=doc_id,
            filename="test.md", chunk_index=i, text=f"文档内容第 {i} 段",
            page_number=i, chunk_strategy="sentence",
        )
        for i in range(1, n + 1)
    ]


@pytest.fixture
def svc(tmp_path) -> IndexService:
    return IndexService(
        FakeEmbedder(), strategy="sentence", index_dir=tmp_path, dim=8
    )


class TestAddRecords:
    @pytest.mark.asyncio
    async def test_add_and_size(self, svc):
        added = await svc.add_records(make_records(5))
        assert added == 5
        assert svc.size == 5
        assert svc.ready is True

    @pytest.mark.asyncio
    async def test_empty_returns_zero(self, svc):
        assert await svc.add_records([]) == 0
        assert svc.size == 0
        assert svc.ready is False

    @pytest.mark.asyncio
    async def test_duplicate_ids_skipped(self, svc):
        """
        重复 id 必须跳过 —— FAISS 的 add_with_ids 对重复 id 不报错，
        只会追加新记录，导致两条不同向量共享同一 id。
        这是静默数据损坏，必须由本层防护。
        """
        records = make_records(3)
        assert await svc.add_records(records) == 3
        assert await svc.add_records(records) == 0, "同批重复 id 应全部跳过"
        assert svc.size == 3, "索引不应增长"


class TestPersistence:
    @pytest.mark.asyncio
    async def test_save_load_roundtrip(self, svc, tmp_path):
        records = make_records(4)
        await svc.add_records(records)

        # 新实例从磁盘加载
        restored = IndexService(
            FakeEmbedder(), strategy="sentence", index_dir=tmp_path, dim=8
        )
        restored.load()

        assert restored.size == 4
        assert len(restored.all_records()) == 4
        for r in records:
            got = restored.record_of(r.faiss_id)
            assert got is not None
            assert got.text == r.text
            assert got.chunk_id == r.chunk_id
            assert got.page_number == r.page_number

    @pytest.mark.asyncio
    async def test_load_missing_files_is_graceful(self, tmp_path):
        """索引不存在时应保持空并可用，不抛异常（否则应用起不来）"""
        empty = IndexService(
            FakeEmbedder(), strategy="nonexistent", index_dir=tmp_path, dim=8
        )
        empty.load()
        assert empty.size == 0
        assert empty.ready is False

    @pytest.mark.asyncio
    async def test_rebuild_is_idempotent(self, svc):
        records = make_records(5)
        vectors = [FakeEmbedder()._vec(r.text) for r in records]

        assert svc.rebuild(records, vectors) == 5
        assert svc.size == 5
        # 再跑一次不应翻倍
        assert svc.rebuild(records, vectors) == 5
        assert svc.size == 5, "rebuild 必须幂等"

    def test_rebuild_length_mismatch_raises(self, svc):
        with pytest.raises(ValueError):
            svc.rebuild(make_records(3), [[0.0] * 8])


class TestSearch:
    @pytest.mark.asyncio
    async def test_finds_most_similar(self, svc):
        await svc.add_records(make_records(5))
        hits = await svc.search("文档内容第 3 段", retrieve_k=5, final_k=3)
        assert len(hits) == 3
        assert hits[0].record.text == "文档内容第 3 段"
        assert hits[0].score > hits[1].score, "必须按相似度降序"

    @pytest.mark.asyncio
    async def test_empty_index_returns_empty(self, svc):
        assert await svc.search("任意查询") == []

    @pytest.mark.asyncio
    async def test_min_score_filters(self, svc):
        await svc.add_records(make_records(5))
        strict = await svc.search("文档内容第 1 段", retrieve_k=5, min_score=0.99, final_k=5)
        assert len(strict) <= 1

    @pytest.mark.asyncio
    async def test_final_k_limits(self, svc):
        await svc.add_records(make_records(10))
        hits = await svc.search("文档内容", retrieve_k=10, final_k=2)
        assert len(hits) == 2

    @pytest.mark.asyncio
    async def test_document_filter(self, svc):
        await svc.add_records(make_records(3, doc_id="doc-A"))
        await svc.add_records([
            IndexRecord(faiss_id=10 + i, chunk_id=f"c{10+i}", document_id="doc-B",
                        filename="b.md", chunk_index=i, text=f"另一文档内容 {i}")
            for i in range(3)
        ])
        hits = await svc.search("内容", retrieve_k=10, final_k=10, document_id="doc-B")
        assert hits, "应能检索到 doc-B"
        assert all(h.record.document_id == "doc-B" for h in hits)


class TestRemoveDocument:
    @pytest.mark.asyncio
    async def test_removes_only_target_document(self, svc):
        await svc.add_records(make_records(3, doc_id="keep"))
        await svc.add_records([
            IndexRecord(faiss_id=10 + i, chunk_id=f"c{10+i}", document_id="drop",
                        filename="d.md", chunk_index=i, text=f"待删内容 {i}")
            for i in range(2)
        ])
        assert svc.size == 5

        removed = svc.remove_document("drop")

        assert removed == 2
        assert svc.size == 3
        assert all(r.document_id == "keep" for r in svc.all_records())

    @pytest.mark.asyncio
    async def test_remove_nonexistent_returns_zero(self, svc):
        await svc.add_records(make_records(2))
        assert svc.remove_document("不存在") == 0
        assert svc.size == 2


class TestLookup:
    @pytest.mark.asyncio
    async def test_get_by_chunk_id(self, svc):
        await svc.add_records(make_records(3))
        assert svc.get_by_chunk_id("chunk-2") is not None
        assert svc.get_by_chunk_id("chunk-999") is None

    @pytest.mark.asyncio
    async def test_stats(self, svc):
        await svc.add_records(make_records(3))
        s = svc.stats()
        assert s["strategy"] == "sentence"
        assert s["vectors"] == 3
        assert s["records"] == 3
        assert s["documents"] == 1
        assert s["exists"] is True

    def test_stats_on_empty(self, svc):
        s = svc.stats()
        assert s["vectors"] == 0
        assert s["exists"] is False
