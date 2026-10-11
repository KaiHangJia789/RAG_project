"""
MCP Server — 把 RAG 知识库通过 Model Context Protocol 暴露给外部 AI 客户端

**保持本文件 import-light**：只写 docstring，不放任何 import 或代码。
原因：`python -m app.mcp_server` 会先 import 这个包，而 stdio 模式下
**任何在 serving 开始之前写到 stdout 的内容都会污染 JSON-RPC 通道**。
包级别的重量级 import 或副作用是在 serving 窗口之外的，SDK 的 stdout
重定向保护不到。

MCP 的三层原语对应关系：
  - Tool     → search_documents / get_document_by_id（模型主动调用）
  - Resource → docs://list（客户端读取的数据）
  - Prompt   → rag_qa（预置提示模板）
"""
