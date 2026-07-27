-- ============================================================
-- 002_documents.sql
-- 文档表 — 上传文件元信息
-- 幂等：可安全重复执行
-- ============================================================

-- 文档状态枚举（DO 块做幂等保护）
DO $$
BEGIN
    CREATE TYPE document_status AS ENUM (
        'uploaded', 'parsing', 'chunking', 'ready', 'failed'
    );
EXCEPTION
    WHEN duplicate_object THEN
        RAISE NOTICE '类型 document_status 已存在，跳过';
END;
$$;

CREATE TABLE IF NOT EXISTS documents (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID           NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename     VARCHAR(512)   NOT NULL,
    file_type    VARCHAR(16)    NOT NULL,
    file_size    BIGINT         NOT NULL CHECK (file_size >= 0),
    storage_path VARCHAR(1024)  NOT NULL,
    status       document_status NOT NULL DEFAULT 'uploaded',
    created_at   TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ    NOT NULL DEFAULT NOW()
);

-- ===== 索引（按查询场景分阶段启用） =====

-- Week5 核心索引 1: 用户文档 Keyset 分页
--   SQL: WHERE user_id = $1 ORDER BY created_at DESC, id DESC
CREATE INDEX IF NOT EXISTS idx_documents_user_keyset
    ON documents (user_id, created_at DESC, id DESC);

-- Week5 核心索引 2: FK 加速（JOIN users）
CREATE INDEX IF NOT EXISTS idx_documents_user_id
    ON documents (user_id);

-- Week5 核心索引 3: 文件名搜索（pg_trgm 模糊匹配）
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX IF NOT EXISTS idx_documents_filename_trgm
    ON documents USING gin (filename gin_trgm_ops);

-- Week9 按需启用: 全量 Keyset 分页（管理员场景）
-- CREATE INDEX IF NOT EXISTS idx_documents_keyset
--     ON documents (created_at DESC, id DESC);

-- Week9 按需启用: 按状态过滤的分页
-- CREATE INDEX IF NOT EXISTS idx_documents_status_keyset
--     ON documents (status, created_at DESC, id DESC)
--     WHERE status = 'ready';
