"""
Pytest 配置文件
提供测试客户端、fixtures 和内存测试替身（DB + Cache）
"""
from contextlib import asynccontextmanager
from datetime import datetime, UTC

import pytest
from httpx import ASGITransport, AsyncClient

from app.cache.document_cache import DocumentCache
from app.db.repositories.document_repo import DocumentRepository
from app.services.document_service import DocumentService
from app.services.storage_service import StorageService


# ═══════════════════════════════════════════════════════════════
# 内存 DB 测试替身（模拟 asyncpg 连接池 + 事务）
# ═══════════════════════════════════════════════════════════════

class FakeConnection:
    """模拟 asyncpg.Connection — 轻量但完整的 CRUD 实现"""

    def __init__(self, store: dict[str, dict]):
        self._store = store

    async def fetchrow(self, query: str, *args):
        """模拟单行查询/插入/删除"""
        sql = query.strip()

        # INSERT ... RETURNING *
        if sql.upper().startswith("INSERT"):
            # 从 SQL 提取列名: INSERT INTO "table" ("col1","col2",...) VALUES ($1,$2,...) RETURNING *
            import re, uuid
            cols_match = re.search(r'\(([^)]+)\)\s*VALUES', sql)
            if cols_match:
                cols = [c.strip().strip('"') for c in cols_match.group(1).split(",")]
                row = dict(zip(cols, args))
                if "id" not in row:
                    row["id"] = str(uuid.uuid4())
                self._store[row["id"]] = row
                return self._Record(row)
            return None

        # DELETE ... RETURNING id
        if sql.upper().startswith("DELETE"):
            row_id = str(args[-1])  # WHERE id = $N → 最后一个参数
            if row_id in self._store:
                del self._store[row_id]
                return self._Record({"id": row_id})
            return None

        # UPDATE ... RETURNING *
        if sql.upper().startswith("UPDATE"):
            row_id = str(args[-1])  # WHERE id = $N → 最后一个参数
            if row_id in self._store:
                # 解析 SET 子句: col = $1, col2 = $2
                import re
                set_match = re.search(r'SET\s+(.+?)\s+WHERE', sql, re.IGNORECASE)
                if set_match:
                    set_parts = [p.strip().split("=")[0].strip().strip('"') for p in set_match.group(1).split(",")]
                    for i, col in enumerate(set_parts):
                        self._store[row_id][col] = args[i]
                self._store[row_id]["updated_at"] = row_id  # 至少更新一下
                return self._Record(self._store[row_id])
            return None

        # SELECT ... (包括 COUNT)
        if sql.upper().startswith("SELECT"):
            # COUNT(*) → 支持 user_id 过滤
            if "COUNT(*)" in sql.upper():
                user_id = str(args[0]) if args else None
                if user_id:
                    count = sum(1 for v in self._store.values() if v.get("user_id") == user_id)
                else:
                    count = len(self._store)
                return self._Record({"count": count})

            # SELECT * FROM ... WHERE id = $1
            # 提取 WHERE id = 后面的参数
            import re
            where_match = re.search(r'WHERE\s+.+?\$(\d+)', sql)
            if where_match:
                param_idx = int(where_match.group(1)) - 1
                if param_idx < len(args):
                    row_id = str(args[param_idx])
                    if row_id in self._store:
                        return self._Record(self._store[row_id])
            # SELECT * FROM ... WHERE user_id = $1 ...
            if "user_id" in sql.lower() and args:
                # 返回全部匹配用户ID的行（简化实现）
                user_id = str(args[0])
                rows = [v for v in self._store.values() if v.get("user_id") == user_id]
                if rows:
                    return self._Record(rows[0])

        return None

    async def fetch(self, query: str, *args):
        """模拟多行查询 — 支持 SELECT, 过滤, 排序, LIMIT"""
        sql = query.upper()

        # COUNT(*)
        if "COUNT(*)" in sql:
            user_id = str(args[0]) if args else None
            if user_id:
                count = sum(1 for v in self._store.values() if v.get("user_id") == user_id)
            else:
                count = len(self._store)
            return [self._Record({"count": count})]

        # 通用 SELECT * 处理
        if "SELECT" in sql and "FROM" in sql:
            rows = list(self._store.values())

            # 提取 WHERE 过滤条件
            import re
            # 匹配 user_id = $1
            user_filter = re.search(r'user_id\s*=\s*\$1', sql)
            if user_filter and args:
                user_id = str(args[0])
                rows = [r for r in rows if r.get("user_id") == user_id]

            # 处理 LIMIT ($N 参数)
            limit_match = re.search(r'LIMIT\s+\$(\d+)', sql, re.IGNORECASE)
            if limit_match:
                param_idx = int(limit_match.group(1)) - 1
                if param_idx < len(args):
                    limit = int(args[param_idx])
                    rows = rows[:limit]

            return [self._Record(r) for r in rows]

        return []

    async def execute(self, query: str, *args):
        return "OK"

    def quote_ident(self, name: str) -> str:
        return f'"{name}"'

    def transaction(self):
        """asyncpg 的事务上下文 — 这里用 no-op 模拟"""
        class _FakeTx:
            async def __aenter__(self): return None
            async def __aexit__(self, *args): pass
        return _FakeTx()

    class _Record:
        """模拟 asyncpg.Record — 支持键名和数字索引"""
        def __init__(self, data: dict):
            self._data = data
            self._keys = list(data.keys())

        def __getitem__(self, key):
            if isinstance(key, int):
                return list(self._data.values())[key]
            return self._data[key]

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

        def keys(self):
            return self._data.keys()

        def values(self):
            return self._data.values()

        def items(self):
            return self._data.items()

        def get(self, key, default=None):
            if isinstance(key, int):
                vals = list(self._data.values())
                return vals[key] if key < len(vals) else default
            return self._data.get(key, default)

        def __repr__(self):
            return f"Record({self._data})"

        def dict(self):
            return dict(self._data)


class FakePool:
    """模拟 asyncpg.Pool"""

    def __init__(self, store: dict[str, dict]):
        self._store = store

    def acquire(self):
        class _Ctx:
            def __init__(self, store):
                self.conn = FakeConnection(store)
            async def __aenter__(self):
                return self.conn
            async def __aexit__(self, *args):
                pass
        return _Ctx(self._store)


class FakeDB:
    """模拟 Database — 所有数据存内存 dict"""

    def __init__(self):
        self.store: dict[str, dict] = {}
        self.pool = FakePool(self.store)

    async def connect(self, dsn): pass
    async def disconnect(self): pass

    async def fetch_one(self, query: str, *args):
        conn = FakeConnection(self.store)
        return await conn.fetchrow(query, *args)

    async def fetch_all(self, query: str, *args):
        conn = FakeConnection(self.store)
        return await conn.fetch(query, *args)

    async def execute(self, query: str, *args):
        return "OK"

    @asynccontextmanager
    async def transaction(self):
        conn = FakeConnection(self.store)
        yield conn


# ═══════════════════════════════════════════════════════════════
# 内存 Cache 测试替身（模拟 Redis）
# ═══════════════════════════════════════════════════════════════

class FakeRedis:
    """模拟 redis.asyncio.Redis — 轻量内存实现"""

    def __init__(self):
        self._store: dict[str, dict] = {}
        self._expire: dict[str, float] = {}
        self._sorted_sets: dict[str, dict[str, float]] = {}

    # Hash 操作
    async def hgetall(self, key: str) -> dict:
        return self._store.get(key, {}).copy()

    async def hset(self, key: str, field_or_mapping=None, value=None, mapping=None, **kwargs):
        """支持 hset(key, field, value) 和 hset(key, mapping={...}) 两种签名"""
        if key not in self._store:
            self._store[key] = {}
        if mapping:
            self._store[key].update(mapping)
        elif value is not None and field_or_mapping is not None:
            # hset(key, field, value)
            self._store[key][field_or_mapping] = value
        elif field_or_mapping is not None and isinstance(field_or_mapping, dict):
            # hset(key, mapping_dict) — 第二个位置参数是 dict
            self._store[key].update(field_or_mapping)
        self._store[key].update(kwargs)

    async def expire(self, key: str, ttl: int):
        import time
        self._expire[key] = time.monotonic() + ttl

    # String 操作
    async def set(self, key: str, value: str, nx: bool = False, ex: int = None):
        if nx and key in self._store:
            return False
        self._store.setdefault(key, {})["__value__"] = value
        if ex:
            import time
            self._expire[key] = time.monotonic() + ex
        return True

    async def get(self, key: str) -> str | None:
        return self._store.get(key, {}).get("__value__")

    # 通用
    async def delete(self, *keys: str):
        for k in keys:
            self._store.pop(k, None)
            self._sorted_sets.pop(k, None)

    async def ping(self):
        return True

    async def aclose(self):
        pass

    # Sorted Set
    async def zincrby(self, key: str, amount: float, member: str):
        if key not in self._sorted_sets:
            self._sorted_sets[key] = {}
        self._sorted_sets[key][member] = self._sorted_sets[key].get(member, 0) + amount

    async def zrevrange(self, key: str, start: int, stop: int, withscores: bool = False):
        if key not in self._sorted_sets:
            return []
        sorted_items = sorted(
            self._sorted_sets[key].items(), key=lambda x: x[1], reverse=True
        )
        result = sorted_items[start:stop + 1]
        if withscores:
            return [(member, score) for member, score in result]
        return [member for member, _ in result]

    async def zrem(self, key: str, *members: str):
        if key in self._sorted_sets:
            for m in members:
                self._sorted_sets[key].pop(m, None)

    # Pipeline
    def pipeline(self):
        return _FakePipeline(self)

    @staticmethod
    def from_url(url: str, **kwargs):
        return FakeRedis()


class _FakePipeline:
    """模拟 redis Pipeline — 同步记录命令，__aexit__ 时批量执行"""
    def __init__(self, redis: FakeRedis):
        self._redis = redis
        self._commands = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        for name, args, kwargs in self._commands:
            method = getattr(self._redis, name, None)
            if method:
                await method(*args, **kwargs)
        self._commands.clear()

    def __getattr__(self, name):
        # Pipeline 命令是同步的：记录命令并返回 self 以支持链式调用
        def cmd(*args, **kwargs):
            self._commands.append((name, args, kwargs))
            return self
        return cmd


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def fake_db():
    """内存 DB 测试替身"""
    return FakeDB()


@pytest.fixture
def fake_redis():
    """内存 Redis 测试替身"""
    return FakeRedis()


@pytest.fixture
def storage_service(tmp_path):
    """使用临时目录的存储服务"""
    from app.config import settings
    original = settings.UPLOAD_DIR
    settings.UPLOAD_DIR = tmp_path / "uploads"
    service = StorageService()
    yield service
    settings.UPLOAD_DIR = original


@pytest.fixture
def document_service(fake_db, fake_redis, storage_service):
    """完整 DocumentService — 使用内存 DB + Cache + 真实文件存储"""
    from app.cache.document_cache import DocumentCache
    from app.db.repositories.document_repo import DocumentRepository

    cache = DocumentCache(fake_redis)
    repo = DocumentRepository()
    return DocumentService(
        db=fake_db,
        cache=cache,
        storage=storage_service,
        doc_repo=repo,
    )


@pytest.fixture
async def async_client(fake_db, fake_redis, storage_service):
    """异步 HTTP 测试客户端 — 注入内存替身"""
    from app.dependencies import auth
    from app.main import app
    from app.cache.document_cache import DocumentCache
    from app.db.repositories.document_repo import DocumentRepository

    from app.services.parsing_service import ParsingService
    from app.parsing.chunk_splitter import ChunkSplitter, ChunkingConfig

    # 注入测试替身
    auth._db = fake_db
    auth._redis_client = FakeRedisClientAdapter(fake_redis)
    auth._doc_cache = DocumentCache(fake_redis)
    auth._doc_service = DocumentService(
        db=fake_db,
        cache=auth._doc_cache,
        storage=storage_service,
        doc_repo=DocumentRepository(),
    )
    auth._parsing_service = ParsingService(
        db=fake_db,
        splitter=ChunkSplitter(ChunkingConfig(merge_short_threshold=0)),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


class FakeRedisClientAdapter:
    """适配 FakeRedis 给 RedisClient 接口"""
    def __init__(self, fake_redis: FakeRedis):
        self.client = fake_redis
