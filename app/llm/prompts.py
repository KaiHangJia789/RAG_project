"""
Prompt 模板库

集中管理 Prompt，支持占位符渲染。这是"Prompt 模板库 v1"的起点
（Week 9 要求 ≥10 个场景，本周先建 ≥6 组实验素材）。
"""
from pydantic import BaseModel, Field


class PromptTemplate(BaseModel):
    """Prompt 模板"""
    name: str
    description: str
    system: str
    user_template: str                              # 含 {placeholder}

    def render(self, **kwargs) -> tuple[str, str]:
        """渲染 → (system, user_message)"""
        return self.system, self.user_template.format(**kwargs)


# ═══════════════════════════════════════════════════════════════
# 预置模板（≥6 组实验素材）
# ═══════════════════════════════════════════════════════════════

PROMPT_REGISTRY: dict[str, PromptTemplate] = {
    "baseline": PromptTemplate(
        name="baseline",
        description="基线：直接提问，无任何约束",
        system="",
        user_template="{input}",
    ),

    "structured": PromptTemplate(
        name="structured",
        description="结构化：角色 + 约束 + 输出格式",
        system=(
            "你是一个专业的 AI 工程师。回答问题时：\n"
            "1. 先给出核心结论（一句话）\n"
            "2. 再分点说明要点\n"
            "3. 最后给出一个可操作的示例"
        ),
        user_template="{input}",
    ),

    "concise": PromptTemplate(
        name="concise",
        description="精简：强制限定字数",
        system="你是一个回答简洁的助手。回答不得超过 100 字。",
        user_template="{input}",
    ),

    "json_output": PromptTemplate(
        name="json_output",
        description="JSON 输出：要求结构化 JSON 格式",
        system=(
            "你是一个信息提取助手。请始终以 JSON 格式输出，"
            "字段为：summary（摘要）、key_points（要点列表）、conclusion（结论）。"
        ),
        user_template="{input}",
    ),

    "chain_of_thought": PromptTemplate(
        name="chain_of_thought",
        description="思维链：要求分步推理",
        system=(
            "你是一个严谨的分析助手。回答前先分步推理：\n"
            "1. 理解问题\n"
            "2. 拆解关键概念\n"
            "3. 逐步推导\n"
            "4. 给出最终答案"
        ),
        user_template="{input}",
    ),

    "few_shot": PromptTemplate(
        name="few_shot",
        description="少样本：给一个示例再回答",
        system=(
            "你是一个技术问答助手。参考下面的示例格式回答：\n\n"
            "示例问题：什么是向量数据库？\n"
            "示例回答：向量数据库是专门存储和检索高维向量的数据库，"
            "它通过近似最近邻搜索（ANN）算法，在海量向量中快速找到最相似的条目。"
        ),
        user_template="{input}",
    ),

    "query_rewrite": PromptTemplate(
        name="query_rewrite",
        description="多轮改写：把口语化/含指代的问题改写为检索友好的独立 query",
        system=(
            "你是一个查询改写助手。把用户的原始问题改写成一个独立、完整、"
            "适合向量检索的查询语句。要求：\n"
            "1. 补全所有指代（如'它'、'这个'要替换成具体对象）\n"
            "2. 保留核心语义和关键词\n"
            "3. 只输出改写后的查询，不要解释"
        ),
        user_template="原始问题：{input}\n改写后的查询：",
    ),

    "rag_context": PromptTemplate(
        name="rag_context",
        description="检索增强：给定上下文生成答案",
        system=(
            "你是一个基于检索的问答助手。请仅根据下面提供的上下文回答问题。\n"
            "要求：\n"
            "1. 只使用上下文中的信息，不要编造\n"
            "2. 如果上下文不足以回答，明确说'根据提供的资料无法回答'\n"
            "3. 回答末尾标注引用的上下文编号（如 [1][2]）"
        ),
        user_template="上下文：\n{context}\n\n问题：{input}",
    ),

    "refuse": PromptTemplate(
        name="refuse",
        description="拒答：上下文中无答案时明确拒绝，不编造",
        system=(
            "你是一个严谨的问答助手。你的原则是：宁可说不知道，也不编造。\n"
            "当且仅当你能基于可靠信息回答时，才给出答案；"
            "否则明确说'我无法确定'并说明原因。"
        ),
        user_template="{input}",
    ),

    "code_gen": PromptTemplate(
        name="code_gen",
        description="代码生成：带语言和风格约束",
        system=(
            "你是一个严谨的软件工程师。写代码时遵循：\n"
            "1. 只输出代码和必要注释，不解释废话\n"
            "2. 代码风格干净、命名清晰、可运行\n"
            "3. 考虑边界条件和错误处理"
        ),
        user_template="语言：{language}\n任务：{input}",
    ),
}
