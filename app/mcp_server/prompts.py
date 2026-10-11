"""
MCP Prompt 注册

**Prompt 是 MCP 三层原语里最容易被忽略的一层**。
它不执行任何逻辑，只是给客户端提供**预置的提示模板** ——
用户在 Claude Desktop 里可以通过斜杠命令或菜单直接选用。

它把前两层串联起来：提示词本身可以引导模型先调 Tool 检索、
再用 Resource 确认文档范围。这是「用协议表达工作流」的体现。
"""
import logging

logger = logging.getLogger("rag_api.mcp.prompts")


def register_prompts(mcp) -> None:
    """注册 MCP Prompt"""

    @mcp.prompt(
        name="rag_qa",
        title="知识库问答",
        description="基于知识库回答问题，要求先检索再作答并标注来源。",
    )
    def rag_qa(question: str, style: str = "简洁") -> str:
        """
        生成 RAG 问答提示。

        Args:
            question: 要问的问题
            style: 回答风格（简洁 / 详细 / 分点）
        """
        return (
            f"请基于知识库回答这个问题：{question}\n"
            f"\n"
            f"要求：\n"
            f"1. 先用 search_documents 检索相关内容，不要凭已有知识作答\n"
            f"2. 如果检索结果不完整，用 get_document_by_id 读取原文补充\n"
            f"3. 以「{style}」的方式回答\n"
            f"4. 每个事实性陈述标注来源（文档名 + 页码）\n"
            f"5. 如果知识库里确实没有答案，明确说「知识库中没有相关内容」"
        )

    @mcp.prompt(
        name="summarize_document",
        title="文档摘要",
        description="通读指定文档并生成结构化摘要。",
    )
    def summarize_document(document_id: str, focus: str = "全部要点") -> str:
        """
        生成文档摘要提示。

        Args:
            document_id: 文档 UUID（可用 docs://list 查询）
            focus: 摘要聚焦方向
        """
        return (
            f"请通读文档 {document_id} 并生成摘要。\n"
            f"\n"
            f"步骤：\n"
            f"1. 先用 docs://list 确认这份文档的分块总数\n"
            f"2. 用 get_document_by_id 分页读取全文（注意跟随返回的 offset 提示）\n"
            f"3. 聚焦「{focus}」生成摘要\n"
            f"4. 摘要按章节组织，标注对应的分块序号以便溯源"
        )

    logger.debug("已注册 MCP 提示: rag_qa, summarize_document")
