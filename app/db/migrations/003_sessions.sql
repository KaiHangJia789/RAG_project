-- ============================================================
-- 003_sessions.sql
-- 会话表 — 用户对话历史（JSONB 存储消息列表）
-- 幂等：可安全重复执行
-- ============================================================

CREATE TABLE IF NOT EXISTS sessions (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    UUID          NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title      VARCHAR(256)  NOT NULL DEFAULT '未命名会话',
    messages   JSONB         NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

-- 等值列在前 → 范围列在后
CREATE INDEX IF NOT EXISTS idx_sessions_user_keyset
    ON sessions (user_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id
    ON sessions (user_id);
