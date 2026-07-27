-- ============================================================
-- 004_chunks.sql
-- 文档分块表 — Week5 预建，Week9 启用向量检索
-- 幂等：可安全重复执行
-- ============================================================

-- 1. pgvector 扩展（只在镜像支持时安装）
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS vector;
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'pgvector 扩展不可用，跳过向量列创建（请使用 pgvector/pgvector:pg16 镜像）';
END;
$$;

-- 2. 建表
CREATE TABLE IF NOT EXISTS chunks (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  UUID          NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INT           NOT NULL,
    chunk_text   TEXT          NOT NULL,
    chunk_hash   VARCHAR(64)   NOT NULL,
    token_count  INT,
    page_number  INT,
    created_at   TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_chunks_doc_index UNIQUE (document_id, chunk_index)
);

-- 3. 复合索引（Week5 即刻启用）
CREATE INDEX IF NOT EXISTS idx_chunks_doc_keyset
    ON chunks (document_id, chunk_index);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id
    ON chunks (document_id);

-- 4. 向量列 — 直接执行 + EXCEPTION 捕获（防并发竞争）
DO $$
BEGIN
    EXECUTE 'ALTER TABLE chunks ADD COLUMN embedding vector(1536)';
EXCEPTION
    WHEN duplicate_column THEN
        RAISE NOTICE '向量列 embedding 已存在，跳过';
    WHEN others THEN
        RAISE NOTICE '向量列创建失败（pgvector 扩展可能未安装）: %', SQLERRM;
END;
$$;

-- 5. 向量索引 — 仅 pgvector 可用且列存在时创建
DO $$
BEGIN
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_chunks_embedding
             ON chunks USING ivfflat (embedding vector_cosine_ops)
             WHERE embedding IS NOT NULL';
EXCEPTION
    WHEN undefined_object THEN
        RAISE NOTICE '向量索引跳过 — pgvector 或列不可用';
    WHEN others THEN
        RAISE NOTICE '向量索引创建失败: %', SQLERRM;
END;
$$;
