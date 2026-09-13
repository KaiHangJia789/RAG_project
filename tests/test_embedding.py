"""
Embedding 客户端测试（mock API，不真实调用）
"""
import pytest

from app.embedding import DashscopeEmbeddingClient


class FakeData:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding


class FakeResponse:
    def __init__(self, data):
        self.data = data


class FakeEmbeddings:
    """模拟 openai client.embeddings"""

    def __init__(self, api):
        self._api = api

    async def create(self, **kwargs):
        self._api.calls.append(kwargs)
        inputs = kwargs["input"]
        if isinstance(inputs, str):
            inputs = [inputs]
        data = [
            FakeData(index=i, embedding=[0.5] * self._api.dim)
            for i in range(len(inputs))
        ]
        return FakeResponse(data)


class FakeEmbeddingAPI:
    """模拟整个 openai 客户端"""

    def __init__(self, dim=1024):
        self.dim = dim
        self.calls = []
        self.embeddings = FakeEmbeddings(self)


@pytest.fixture
def fake_api():
    return FakeEmbeddingAPI()


@pytest.fixture
def client(fake_api):
    c = DashscopeEmbeddingClient(batch_size=2, max_retries=1)
    c._client = fake_api
    return c


class TestEmbeddingClient:
    @pytest.mark.asyncio
    async def test_embed_empty(self, client):
        assert await client.embed([]) == []

    @pytest.mark.asyncio
    async def test_embed_basic(self, client):
        vecs = await client.embed(["文本一", "文本二"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 1024

    @pytest.mark.asyncio
    async def test_embed_one(self, client):
        vec = await client.embed_one("单条")
        assert len(vec) == 1024

    @pytest.mark.asyncio
    async def test_batching(self, client, fake_api):
        """batch_size=2，输入 5 条 → 应分 3 批调用"""
        await client.embed(["a", "b", "c", "d", "e"])
        assert len(fake_api.calls) == 3

    @pytest.mark.asyncio
    async def test_dimension_mismatch(self, client):
        """返回维度不一致 → 抛异常（安全措施）"""
        client._client.dim = 512  # 篡改为错误维度
        with pytest.raises(ValueError):
            await client.embed(["测试"])

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, fake_api):
        """网络失败应重试"""
        client = DashscopeEmbeddingClient(batch_size=2, max_retries=3)
        client._client = fake_api

        # 第一次抛异常，之后成功
        original_create = fake_api.embeddings.create
        call_count = [0]

        async def flaky_create(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("网络抖动")
            return await original_create(**kwargs)

        fake_api.embeddings.create = flaky_create
        vecs = await client.embed(["测试"])
        assert len(vecs) == 1
        assert call_count[0] == 2  # 失败 1 次 + 重试成功 1 次
