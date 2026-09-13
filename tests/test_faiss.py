"""
FAISS 索引测试
"""
import pytest

from app.vector import FaissIndex


class TestFaissIndex:
    def test_add_and_search(self):
        index = FaissIndex(dim=4)
        # 添加两个正交向量（归一化后）
        index.add([[1, 0, 0, 0], [0, 1, 0, 0]])
        assert len(index) == 2

        results = index.search([1, 0, 0, 0], top_k=2)
        # 第一个结果应该是 id=0（与查询最相似）
        assert results[0][0] == 0
        assert results[0][1] > results[1][1]

    def test_dimension_mismatch(self):
        index = FaissIndex(dim=4)
        with pytest.raises(ValueError):
            index.add([[1, 2, 3]])  # 3 维，索引是 4 维

    def test_top_k_boundary(self):
        index = FaissIndex(dim=4)
        index.add([[1, 0, 0, 0], [0, 1, 0, 0]])
        # top_k > 索引大小 → 只返回实际数量
        results = index.search([1, 0, 0, 0], top_k=10)
        assert len(results) == 2

    def test_empty_index_search(self):
        index = FaissIndex(dim=4)
        assert index.search([1, 0, 0, 0], top_k=5) == []

    def test_invalid_top_k(self):
        index = FaissIndex(dim=4)
        index.add([[1, 0, 0, 0]])
        with pytest.raises(ValueError):
            index.search([1, 0, 0, 0], top_k=0)

    def test_custom_ids(self):
        index = FaissIndex(dim=4)
        index.add([[1, 0, 0, 0], [0, 1, 0, 0]], ids=[100, 200])
        results = index.search([1, 0, 0, 0], top_k=1)
        assert results[0][0] == 100

    def test_save_load(self, tmp_path):
        index = FaissIndex(dim=4)
        index.add([[1, 0, 0, 0], [0, 1, 0, 0]], ids=[1, 2])

        path = tmp_path / "test.index"
        index.save(path)

        loaded = FaissIndex.load(path, dim=4)
        assert len(loaded) == 2
        results = loaded.search([1, 0, 0, 0], top_k=1)
        assert results[0][0] == 1

    def test_load_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            FaissIndex.load("nonexistent.index", dim=4)
