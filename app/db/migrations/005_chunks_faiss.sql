-- ============================================================
-- 005_chunks_faiss.sql
-- Week10: chunks 表加 FAISS 向量 id 映射 + chunk 策略维度
-- 幂等：可安全重复执行
--
-- 背景：Week9 的 chunk 三策略只存在于脚本里，入库路径用的是 ChunkSplitter。
-- Week10 要把三策略提升为主路径并做对比，需要：
--   1. faiss_id —— FAISS 要 int64，chunks 主键是 UUID，二者之间没有桥
--   2. chunk_strategy —— 同一文档按不同策略分块，靠此列区分
-- ============================================================

-- ── 1. chunk_strategy 列 ──────────────────────────────────────
-- 顺序固定：加列 → 回填旧数据 → 设默认 → 设 NOT NULL
-- （直接对已有 NULL 行的表加 NOT NULL 会失败）
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunk_strategy VARCHAR(32);

UPDATE chunks SET chunk_strategy = 'splitter' WHERE chunk_strategy IS NULL;

ALTER TABLE chunks ALTER COLUMN chunk_strategy SET DEFAULT 'splitter';
ALTER TABLE chunks ALTER COLUMN chunk_strategy SET NOT NULL;

-- ── 2. faiss_id 列 + sequence ─────────────────────────────────
-- 用 PG sequence 而非应用层计数器：应用层计数器重启后会从 1 重新发号，
-- 而 faiss.IndexIDMap.add_with_ids 对重复 id 不报错（追加新记录），
-- 结果是两条不同向量共享同一 id → 静默数据损坏。nextval 天然无此问题。
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS faiss_id BIGINT;

CREATE SEQUENCE IF NOT EXISTS chunks_faiss_id_seq;
ALTER SEQUENCE chunks_faiss_id_seq OWNED BY chunks.faiss_id;
ALTER TABLE chunks ALTER COLUMN faiss_id SET DEFAULT nextval('chunks_faiss_id_seq');

-- ── 3. 唯一约束替换（关键！）──────────────────────────────────
-- 旧约束 (document_id, chunk_index) 与"多策略对比"不兼容：
-- 三种策略的 chunk_index 都从 0 开始，第 2 种策略插入时必然冲突。
-- DROP + ADD 放同一事务，避免中间态无约束。
DO $$
BEGIN
    ALTER TABLE chunks DROP CONSTRAINT IF EXISTS uq_chunks_doc_index;
    ALTER TABLE chunks DROP CONSTRAINT IF EXISTS uq_chunks_doc_strategy_index;
    ALTER TABLE chunks ADD CONSTRAINT uq_chunks_doc_strategy_index
        UNIQUE (document_id, chunk_strategy, chunk_index);
EXCEPTION
    WHEN others THEN
        RAISE NOTICE '唯一约束替换失败: %', SQLERRM;
        RAISE;
END;
$$;

-- ── 4. faiss_id 唯一部分索引 ──────────────────────────────────
-- 部分索引：faiss_id 允许为 NULL（尚未向量化的 chunk），但非 NULL 值必须唯一
CREATE UNIQUE INDEX IF NOT EXISTS uq_chunks_faiss_id
    ON chunks (faiss_id) WHERE faiss_id IS NOT NULL;

-- ── 5. 策略过滤索引 ───────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_chunks_strategy
    ON chunks (chunk_strategy);
