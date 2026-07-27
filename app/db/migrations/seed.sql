-- ============================================================
-- seed.sql
-- 种子数据 — 开发/测试用初始数据
-- ============================================================

-- 新增用户（冲突时跳过）
INSERT INTO users (username, email, password_hash) VALUES
    ('demo_user', 'demo@example.com',
     '$2b$12$LJ3m4ys3GZfnYMz8kVsKaOTS6KXJ6VvGf7l0Hc1QUoZRPAjdMQeYq')
ON CONFLICT (username) DO NOTHING;

INSERT INTO users (username, email, password_hash) VALUES
    ('test_user', 'test@example.com',
     '$2b$12$LJ3m4ys3GZfnYMz8kVsKaOTS6KXJ6VvGf7l0Hc1QUoZRPAjdMQeYq')
ON CONFLICT (username) DO NOTHING;

-- 新增文档
INSERT INTO documents (user_id, filename, file_type, file_size, storage_path, status)
SELECT u.id, 'RAG技术白皮书.pdf', '.pdf', 204800,
       'uploads/2026/08/doc-white-paper.pdf', 'ready'
FROM users u WHERE u.username = 'demo_user'
AND NOT EXISTS (
    SELECT 1 FROM documents d
    WHERE d.filename = 'RAG技术白皮书.pdf'
       AND d.user_id = u.id
);

INSERT INTO documents (user_id, filename, file_type, file_size, storage_path, status)
SELECT u.id, 'API接口文档.md', '.md', 15360,
       'uploads/2026/08/doc-api-doc.md', 'ready'
FROM users u WHERE u.username = 'demo_user'
AND NOT EXISTS (
    SELECT 1 FROM documents d
    WHERE d.filename = 'API接口文档.md'
       AND d.user_id = u.id
);

INSERT INTO documents (user_id, filename, file_type, file_size, storage_path, status)
SELECT u.id, '测试数据集.csv', '.csv', 4096,
       'uploads/2026/08/doc-test-data.csv', 'uploaded'
FROM users u WHERE u.username = 'demo_user'
AND NOT EXISTS (
    SELECT 1 FROM documents d
    WHERE d.filename = '测试数据集.csv'
       AND d.user_id = u.id
);

-- 新增会话
INSERT INTO sessions (user_id, title, messages)
SELECT u.id, 'RAG架构讨论',
       '[
         {"role":"user","content":"什么是RAG？","timestamp":"2026-08-10T10:00:00Z"},
         {"role":"assistant","content":"RAG（检索增强生成）是结合信息检索与LLM生成的技术...","timestamp":"2026-08-10T10:00:05Z"}
       ]'::jsonb
FROM users u WHERE u.username = 'demo_user'
AND NOT EXISTS (
    SELECT 1 FROM sessions s
    WHERE s.title = 'RAG架构讨论' AND s.user_id = u.id
);
