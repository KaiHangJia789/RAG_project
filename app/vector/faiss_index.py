"""
FAISS 向量索引封装

使用 IndexFlatIP + 余弦归一化（内积 = 余弦相似度），配合 IndexIDMap 存自定义 id。

安全措施:
  - 维度校验：add 时向量维度必须与索引一致，防止错位
  - top_k 边界：search 时 top_k 不超过索引大小，防越界
  - 空索引防护：空索引 search 返回空列表
  - 文件校验：load 时检查文件存在
"""
import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger("rag_api.vector")


class FaissIndex:
    """FAISS 索引封装（精确检索，适合中小规模）"""

    def __init__(self, dim: int):
        if dim <= 0:
            raise ValueError(f"dim 必须 > 0，实际 {dim}")
        self.dim = dim
        # IndexIDMap 支持自定义 id；IndexFlatIP 内积检索
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))

    # ── 写入 ──

    def add(self, vectors: list[list[float]], ids: list[int] | None = None) -> int:
        """
        添加向量。

        Args:
            vectors: 向量列表（每个 dim 维）
            ids: 可选的自定义 id 列表；None 则自动分配 0..n-1

        Returns:
            本次添加的向量数量
        """
        if not vectors:
            return 0

        arr = np.array(vectors, dtype=np.float32)

        # 维度校验（安全措施：防止 FAISS 索引维度错位）
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(
                f"向量维度不匹配：索引期望 {self.dim}，实际 {arr.shape[1] if arr.ndim == 2 else '非二维'}"
            )

        # 余弦归一化（L2 归一化后内积 = 余弦相似度）
        faiss.normalize_L2(arr)

        if ids is None:
            start = self._index.ntotal
            ids = list(range(start, start + len(vectors)))

        if len(ids) != len(vectors):
            raise ValueError(
                f"ids 数量({len(ids)})与向量数量({len(vectors)})不一致"
            )

        self._index.add_with_ids(arr, np.array(ids, dtype=np.int64))
        return len(vectors)

    # ── 检索 ──

    def search(self, query_vec: list[float], top_k: int = 5) -> list[tuple[int, float]]:
        """
        检索 top_k 个最相似的向量。

        Args:
            query_vec: 查询向量（dim 维）
            top_k: 返回数量

        Returns:
            [(id, 相似度分数), ...] 按相似度降序
        """
        if self._index.ntotal == 0:
            return []  # 空索引防护

        if top_k <= 0:
            raise ValueError(f"top_k 必须 > 0，实际 {top_k}")

        # top_k 边界：不超过索引大小
        top_k = min(top_k, self._index.ntotal)

        query = np.array([query_vec], dtype=np.float32)
        if query.shape[1] != self.dim:
            raise ValueError(
                f"查询向量维度不匹配：索引期望 {self.dim}，实际 {query.shape[1]}"
            )
        faiss.normalize_L2(query)

        scores, ids = self._index.search(query, top_k)
        # 过滤掉无效 id（-1 表示未命中）
        result = [
            (int(ids[0][i]), float(scores[0][i]))
            for i in range(top_k)
            if ids[0][i] != -1
        ]
        return result

    # ── 持久化 ──

    def save(self, path: str | Path) -> None:
        """
        保存索引到磁盘。

        用 serialize_index 序列化为 bytes 再用 Python open 写文件，
        彻底绕开 faiss C++ fopen 在 Windows 中文路径下的编码问题。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # serialize_index 返回 numpy.uint8 数组，tobytes() 转 bytes 后用 Python open 写，
        # 绕开 faiss C++ fopen 在 Windows 中文路径下的编码问题
        data = faiss.serialize_index(self._index)
        with open(path, "wb") as f:
            f.write(data.tobytes())
        logger.info("FAISS 索引已保存: %s（%d 条）", path, self._index.ntotal)

    @classmethod
    def load(cls, path: str | Path, dim: int, *, strict: bool = True) -> "FaissIndex":
        """
        从磁盘加载索引。

        Args:
            strict: 维度不一致时是否抛异常。默认 True —— 维度不匹配时 FAISS 不会
                报错，只会返回毫无意义的近邻结果，属于静默故障，必须早失败。
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"FAISS 索引文件不存在: {path}")

        index = cls(dim)
        with open(path, "rb") as f:
            data = f.read()
        # bytes → numpy.uint8 数组（deserialize_index 期望 numpy 数组）
        arr = np.frombuffer(data, dtype=np.uint8)
        index._index = faiss.deserialize_index(arr)

        if index._index.d != dim:
            msg = (
                f"索引文件维度({index._index.d})与配置({dim})不一致。"
                f"多半是 EMBEDDING_MODEL 换过 —— 需删除索引文件重建：{path}"
            )
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
        return index

    # ── 删除 ──

    def remove_ids(self, ids: list[int]) -> int:
        """
        按 id 删除向量，返回实际删除条数。

        删除文档时调用。不删的话索引里会留下指向已删 chunk 的悬空向量，
        检索命中后回表取不到文本。
        """
        if not ids:
            return 0
        before = self._index.ntotal
        self._index.remove_ids(np.array(ids, dtype=np.int64))
        return before - self._index.ntotal

    def ids(self) -> list[int]:
        """返回索引内全部 id（索引一致性自检用）"""
        id_map = faiss.vector_to_array(self._index.id_map)
        return [int(i) for i in id_map]

    def has_id(self, faiss_id: int) -> bool:
        """判断某 id 是否已在索引内（防重复 add 导致 top_k 名额被挤占）"""
        if self._index.ntotal == 0:
            return False
        return bool(np.any(faiss.vector_to_array(self._index.id_map) == faiss_id))

    # ── 属性 ──

    @property
    def size(self) -> int:
        return self._index.ntotal

    def __len__(self) -> int:
        return self._index.ntotal
