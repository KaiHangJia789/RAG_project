# 第5周：数据库与Redis — 持久化与缓存层

**时间**：2026.08.10 — 2026.08.16  
**分支**：`week5-database-redis`  
**前置依赖**：Week4 FastAPI 基础已完成

---

## 一、本周目标

在 Week4 的 FastAPI 应用基础上，将**内存存储**替换为 **PostgreSQL 持久化**，并引入 **Redis 缓存层**，使 RAG 项目具备生产级数据管理能力。

| 维度 | Week4（当前） | Week5（目标） |
|------|:----------:|:----------:|
| 文档存储 | 内存 dict | PostgreSQL |
| 数据一致性 | 无 | ACID 事务 |
| 缓存 | 无 | Redis（会话/热门文档） |
| 数据模型 | 1 张表（Document） | 3 张表（user/document/session） |
| 索引 | 无 | 主键 + 外键 + 复合索引 |
| 可恢复性 | 重启丢失 | 持久化到磁盘 |

---

## 二、学习任务拆解

### 2.1 PostgreSQL 核心知识（Day 1–2）

- [ ] **CRUD 操作**：`INSERT` / `SELECT` / `UPDATE` / `DELETE` 及 `RETURNING` 子句
- [ ] **索引设计**：B-Tree / Hash / GIN 索引的选择与创建时机
- [ ] **事务隔离**：`BEGIN` / `COMMIT` / `ROLLBACK`，理解 ACID
- [ ] **外键约束**：`REFERENCES` + `ON DELETE CASCADE` / `SET NULL`
- [ ] **连接查询**：`INNER JOIN` / `LEFT JOIN` + 子查询
- [ ] **迁移工具**：Alembic 自动版本化管理 Schema 变更

### 2.2 Redis 核心知识（Day 3–4）

- [ ] **String**：缓存 JSON 序列化的文档内容（`GET` / `SETEX`）
- [ ] **Hash**：存储文档元信息的字段级缓存（`HSET` / `HGETALL` / `HDEL`）
- [ ] **List**：最近访问文档队列（`LPUSH` + `LTRIM` 实现 TOP N）
- [ ] **Set**：热门标签/分类去重集合
- [ ] **Sorted Set**：按热度排序的文档排行榜（`ZADD` + `ZREVRANGE`）
- [ ] **过期策略**：`EXPIRE` / `TTL` / `SETEX` 实现自动过期
- [ ] **缓存模式**：Cache-Aside（旁路缓存）/ Write-Through（写穿透）

### 2.3 数据模型设计（Day 5）

- [ ] ER 图绘制（三表关系）
- [ ] SQL 建表脚本（含索引 + 约束）
- [ ] 种子数据脚本（测试用初始数据）

### 2.4 缓存层设计（Day 6）

- [ ] 会话缓存：`session:{session_id}` → JSON，TTL 30min
- [ ] 热门文档缓存：Sorted Set 按访问量排序
- [ ] 缓存失效策略：写操作时主动 invalidate

### 2.5 整合 + 测试（Day 7）

- [ ] DocumentService 从内存版改为 DB 版
- [ ] pytest 测试覆盖 DB + Redis 读写
- [ ] docker-compose 一键启动（app + postgres + redis）

---

## 三、数据模型设计

### 3.1 ER 图（实体关系）

```
┌──────────────┐       ┌──────────────────┐       ┌──────────────┐
│    users     │       │    documents     │       │   sessions   │
├──────────────┤       ├──────────────────┤       ├──────────────┤
│ PK id (UUID) │──┐    │ PK id (UUID)     │   ┌───│ PK id (UUID) │
│ username     │  │    │ FK user_id → user│   │   │ FK user_id   │
│ email        │  │    │ filename         │   │   │ title        │
│ password_hash│  │    │ file_type        │   │   │ messages     │
│ created_at   │  └───>│ file_size        │   │   │ created_at   │
│ updated_at   │       │ storage_path     │   │   │ updated_at   │
└──────────────┘       │ status           │   │   └──────────────┘
                       │ created_at       │   │
                       │ updated_at       │   │
                       └──────────────────┘   │
                                               │
                    ┌──────────────┐           │
                    │ doc_vectors  │           │
                    ├──────────────┤           │
                    │ PK id (UUID) │           │
                    │ FK doc_id ───┘           │
                    │ chunk_index   │
                    │ chunk_text    │
                    │ embedding     │  (Week 9)
                    │ created_at    │
                    └──────────────┘
```

### 3.2 建表 SQL

```sql
-- 001_create_users.sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE users (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username    VARCHAR(64)  NOT NULL UNIQUE,
    email       VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users(email);

-- 002_create_documents.sql
CREATE TYPE document_status AS ENUM (
    'uploaded', 'parsing', 'chunking', 'ready', 'failed'
);

CREATE TABLE documents (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id      UUID           NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename     VARCHAR(512)   NOT NULL,
    file_type    VARCHAR(16)    NOT NULL,
    file_size    BIGINT         NOT NULL CHECK (file_size >= 0),
    storage_path VARCHAR(1024)  NOT NULL,
    status       document_status NOT NULL DEFAULT 'uploaded',
    created_at   TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ    NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_documents_user_id    ON documents(user_id);
CREATE INDEX idx_documents_status     ON documents(status);
CREATE INDEX idx_documents_file_type  ON documents(file_type);
CREATE INDEX idx_documents_created_at ON documents(created_at DESC);
CREATE INDEX idx_documents_filename   ON documents USING gin(filename gin_trgm_ops);

-- 003_create_sessions.sql
CREATE TABLE sessions (
    id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id    UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title      VARCHAR(256) NOT NULL DEFAULT '未命名会话',
    messages   JSONB       NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_sessions_user_id    ON sessions(user_id);
CREATE INDEX idx_sessions_updated_at ON sessions(updated_at DESC);
```

### 3.3 种子数据

```sql
-- seed.sql
INSERT INTO users (username, email, password_hash) VALUES
    ('demo_user', 'demo@example.com', '$2b$12$dummy_hash_for_dev'),
    ('test_user', 'test@example.com', '$2b$12$dummy_hash_for_dev');

INSERT INTO documents (user_id, filename, file_type, file_size, storage_path, status) VALUES
    (
        (SELECT id FROM users WHERE username = 'demo_user'),
        'RAG技术白皮书.pdf', '.pdf', 204800,
        'uploads/2026/08/doc-white-paper.pdf', 'ready'
    ),
    (
        (SELECT id FROM users WHERE username = 'demo_user'),
        'API接口文档.md', '.md', 15360,
        'uploads/2026/08/doc-api-doc.md', 'ready'
    ),
    (
        (SELECT id FROM users WHERE username = 'test_user'),
        '测试数据.csv', '.csv', 4096,
        'uploads/2026/08/doc-test-data.csv', 'uploaded'
    );

INSERT INTO sessions (user_id, title, messages) VALUES
    (
        (SELECT id FROM users WHERE username = 'demo_user'),
        'RAG架构讨论',
        '[
            {"role":"user","content":"什么是RAG？","timestamp":"2026-08-10T10:00:00Z"},
            {"role":"assistant","content":"RAG（检索增强生成）是结合信息检索与LLM生成的技术...","timestamp":"2026-08-10T10:00:05Z"}
        ]'::jsonb
    );
```

---

## 四、Redis 缓存设计

### 4.1 缓存 Key 规范

| 缓存类型 | Key Pattern | 数据类型 | TTL | 说明 |
|---------|------------|---------|-----|------|
| 会话缓存 | `session:{session_id}` | String (JSON) | 30 min | 活跃会话的完整消息历史 |
| 文档元信息 | `doc:meta:{doc_id}` | Hash | 10 min | 文档的字段级缓存 |
| 热门文档 | `doc:hot` | Sorted Set | 持久 | score=访问次数，TOP 20 |
| 最近访问 | `doc:recent:{user_id}` | List | 无（CAP 到 20） | 用户最近访问的文档 ID |
| 用户信息 | `user:{user_id}` | Hash | 15 min | 用户基本信息的字段缓存 |

### 4.2 缓存伪代码

```python
# Cache-Aside 模式：读文档
async def get_document(doc_id: str) -> dict | None:
    # Step 1: 查缓存
    cached = await redis.hgetall(f"doc:meta:{doc_id}")
    if cached:
        return cached  # 缓存命中

    # Step 2: 缓存未命中 → 查数据库
    doc = await db.fetch_one("SELECT * FROM documents WHERE id = $1", doc_id)
    if doc is None:
        return None

    # Step 3: 回写缓存
    await redis.hset(f"doc:meta:{doc_id}", mapping=dict(doc))
    await redis.expire(f"doc:meta:{doc_id}", 600)  # 10分钟

    return dict(doc)

# 写操作：主动 invalidate 缓存
async def update_document(doc_id: str, **updates) -> None:
    await db.execute("UPDATE documents SET ... WHERE id = $1", doc_id)
    await redis.delete(f"doc:meta:{doc_id}")  # 删除缓存，下次读时重建

# 热门文档排行：Sorted Set
async def record_document_access(doc_id: str) -> None:
    await redis.zincrby("doc:hot", 1, doc_id)  # 访问次数 +1

async def get_hot_documents(top_n: int = 20) -> list[str]:
    return await redis.zrevrange("doc:hot", 0, top_n - 1)

# 最近访问：List + LTRIM 实现固定容量
async def record_recent_access(user_id: str, doc_id: str) -> None:
    key = f"doc:recent:{user_id}"
    await redis.lpush(key, doc_id)
    await redis.ltrim(key, 0, 19)  # 只保留最近 20 条
```

### 4.3 缓存失效策略

```python
# 策略1: TTL 自动过期（默认）
#   - 会话缓存: EXPIRE 1800 (30分钟)
#   - 文档缓存: EXPIRE 600  (10分钟)
#   - 用户缓存: EXPIRE 900  (15分钟)

# 策略2: 写操作主动删除
#   - 文档更新 → DELETE doc:meta:{id}
#   - 文档删除 → DELETE doc:meta:{id} + ZREM doc:hot {id}

# 策略3: 缓存穿透防护（空值缓存）
#   - 查询不存在的doc_id → SETEX doc:meta:{id} "NULL" 60
```

---

## 五、项目结构变更

```
app/
├── db/                           # [新增] 数据库模块
│   ├── __init__.py
│   ├── connection.py             # asyncpg 连接池管理
│   ├── migrations/               # SQL 迁移脚本（按版本）
│   │   ├── 001_create_users.sql
│   │   ├── 002_create_documents.sql
│   │   ├── 003_create_sessions.sql
│   │   └── seed.sql
│   └── repositories/             # [新增] 数据访问层 (Repository Pattern)
│       ├── __init__.py
│       ├── user_repo.py
│       ├── document_repo.py
│       └── session_repo.py
├── cache/                        # [新增] 缓存模块
│   ├── __init__.py
│   ├── connection.py             # redis-py 连接管理
│   └── document_cache.py         # 文档缓存操作封装
├── services/
│   └── document_service.py       # [修改] 从内存版 → DB + Cache 版
├── config.py                     # [修改] 添加 DB/Redis 配置
└── ...
```

---

## 六、验收标准 Checklist

### 6.1 PostgreSQL
- [ ] `docker-compose up` 一键启动 postgres + redis + app
- [ ] 三张表成功创建（`\dt` 查看）
- [ ] 种子数据插入成功（`SELECT count(*) FROM documents` ≥ 3）
- [ ] 事务正确回滚：上传失败时数据库无残留记录
- [ ] 删除用户时文档和会话级联删除（`ON DELETE CASCADE`）
- [ ] 索引生效：`EXPLAIN ANALYZE` 显示索引扫描

### 6.2 Redis
- [ ] `SETEX key value TTL` → `TTL key` 返回正确剩余时间
- [ ] 热门文档排行榜：ZADD + ZREVRANGE 输出正确排序
- [ ] 缓存命中：第二次查询同一文档时 trace 显示 "cache hit"
- [ ] 缓存失效：更新文档后缓存 key 被自动删除
- [ ] 会话过期：30 分钟无操作后 session key 自动消失

### 6.3 测试
- [ ] pytest：DB 版本测试覆盖所有 CRUD 操作
- [ ] pytest：Redis 缓存读写 + 过期策略测试
- [ ] 冒烟测试：上传文档 → 查列表 → 查详情 → 删文档 → 确认缓存清理

### 6.4 文档
- [ ] [ER图](er_diagram.md)（或截图）
- [ ] SQL 建表脚本（3 个 migration 文件）
- [ ] [Redis 缓存 Demo 笔记](redis_notes.md)（含过期策略说明）
- [ ] 数据入库验证截图

---

## 七、每日计划

| 日 | 主题 | 产出 |
|----|------|------|
| Day 1 (08.10) | PostgreSQL CRUD 练习 | 个人笔记；`docker-compose.yml` 初版 |
| Day 2 (08.11) | 索引 / 事务 / 外键设计 | 三表建表 SQL 脚本 |
| Day 3 (08.12) | Redis 五种数据结构练习 | Redis 操作笔记 |
| Day 4 (08.13) | 缓存模式 + 过期策略 | 缓存 Key 设计文档 |
| Day 5 (08.14) | Repository 层实现 | `user_repo.py` + `document_repo.py` |
| Day 6 (08.15) | 缓存层实现 + Service 改造 | `document_cache.py` + 改造 `document_service.py` |
| Day 7 (08.16) | 测试 + 验收文档 | 全量 pytest；ER 图；验收 Checklist 全部打勾 |
