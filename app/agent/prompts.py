"""
Agent 专用 Prompt 库

⚠️ **绝不能放进 app/llm/prompts.py 的 PROMPT_REGISTRY**。

原因与判官 prompt 相同：`tests/test_prompts.py::test_all_prompts_renderable`
会遍历业务注册表的**每一个**模板，用固定的 `{input, context, language}` 渲染。
本模块的模板需要 `{question}` / `{context}` / `{query_type}` 等占位符，
混进去会让那个测试直接 KeyError。

设计原则:
  - 全部要求输出 JSON —— 结构化决策（分类、评分）需要可解析的结果
  - 与判官 prompt 一样复用 `PromptTemplate` 类，但不共享注册表
"""
from app.llm.prompts import PromptTemplate

# 换 prompt 必须改版本号（若参与缓存键）
AGENT_PROMPT_VERSION = "agent-v1"


# ═══════════════════════════════════════════════════════════════
# 1. 查询分类（第 12 周条件分支的依据）
# ═══════════════════════════════════════════════════════════════

QUERY_CLASSIFY = PromptTemplate(
    name="agent_query_classify",
    description="把用户问题分类为事实查询/推理查询/闲聊，用于路由到不同检索策略",
    system=(
        "你是一个查询分类助手。把用户的问题分成以下三类之一：\n"
        "\n"
        "- factual（事实查询）：询问具体事实、定义、数值、步骤。"
        "答案通常能在单篇文档的某个段落里直接找到。\n"
        "  例：「什么是 Keyset 分页」「FAISS 的索引类型有哪些」\n"
        "\n"
        "- reasoning（推理查询）：需要综合多处信息、比较、分析或推导。"
        "答案分散在多个段落甚至多篇文档里。\n"
        "  例：「Keyset 和 OFFSET 分页哪个更适合高并发场景，为什么」\n"
        "  例：「如果嵌入模型换了，需要做哪些调整」\n"
        "\n"
        "- chitchat（闲聊）：与知识库内容无关的寒暄、系统功能询问、闲聊。\n"
        "  例：「你好」「你是谁」「今天天气怎么样」\n"
        "\n"
        "判断原则：\n"
        "- 拿不准时选 factual（不要轻易判为 chitchat，那会导致跳过检索）\n"
        "- 只要问题涉及知识库可能覆盖的技术主题，就不是 chitchat\n"
        "\n"
        "只输出 json，不要任何解释，不要 markdown 代码块。\n"
        '输出格式：{"query_type": "factual", "reason": "不超过20字"}'
    ),
    user_template="用户问题：{question}",
)


# ═══════════════════════════════════════════════════════════════
# 2. 检索相关性评分（第 12 周循环重写的依据）
# ═══════════════════════════════════════════════════════════════

RELEVANCE_GRADE = PromptTemplate(
    name="agent_relevance_grade",
    description="评估检索到的上下文能否回答该问题，决定是否需要重写查询",
    system=(
        "你是一个检索质量评估助手。给定一个「问题」和「检索到的上下文」，"
        "判断这些上下文是否**足以回答**该问题。\n"
        "\n"
        "评分标准：\n"
        "- 1.0：上下文直接包含答案，或包含推导答案所需的全部信息\n"
        "- 0.7：上下文包含大部分所需信息，稍作推断即可回答\n"
        "- 0.4：上下文只涉及相关话题，但缺少回答问题的关键信息\n"
        "- 0.0：上下文与问题无关，或完全没有帮助\n"
        "\n"
        "注意：\n"
        "- 判断依据是「能否回答问题」，不是「话题是否相近」。"
        "话题相近但没有给出具体答案 → 低分。\n"
        "- 不要因为上下文里出现了问题中的关键词就给高分。\n"
        "\n"
        "只输出 json，不要任何解释，不要 markdown 代码块。\n"
        '输出格式：{"score": 0.7, "missing": "缺少什么信息，不超过30字"}'
    ),
    user_template="问题：{question}\n\n检索到的上下文：\n{context}",
)


# ═══════════════════════════════════════════════════════════════
# 3. 查询重写（第 12 周循环的修复手段）
# ═══════════════════════════════════════════════════════════════

QUERY_REWRITE = PromptTemplate(
    name="agent_query_rewrite",
    description="当检索质量不达标时，改写查询以提升召回",
    system=(
        "你是一个查询改写助手。上一次检索没有找到足以回答问题的内容，"
        "请改写查询以提升检索命中率。\n"
        "\n"
        "改写策略（按优先级）：\n"
        "1. 如果问题太笼统，加上更具体的技术名词\n"
        "2. 如果问题用了口语化表述，换成文档里可能出现的正式术语\n"
        "3. 如果问题包含指代（「它」「这个」），替换成具体对象\n"
        "4. 如果原查询太长，抽取其中的核心概念重新组合\n"
        "\n"
        "要求：\n"
        "- 保持原问题的核心意图不变，不要改变问题在问什么\n"
        "- 只输出改写后的查询，不要解释\n"
        "- 改写后的查询应该更短或更聚焦，不要堆砌关键词\n"
        "\n"
        "只输出 json，不要任何解释，不要 markdown 代码块。\n"
        '输出格式：{"query": "改写后的查询"}'
    ),
    user_template=(
        "原始问题：{question}\n"
        "上次使用的查询：{last_query}\n"
        "缺少的信息：{missing}\n\n"
        "改写后的查询："
    ),
)


# ═══════════════════════════════════════════════════════════════
# 4. 闲聊直答（第 12 周分支：不检索直接回答）
# ═══════════════════════════════════════════════════════════════

CHITCHAT = PromptTemplate(
    name="agent_chitchat",
    description="闲聊类问题的直答（不经过检索）",
    system=(
        "你是一个知识库问答助手。用户发来的是闲聊或与知识库无关的问题。\n"
        "\n"
        "请简短友好地回应，并说明你擅长的领域：基于已上传的文档回答问题。\n"
        "不要编造知识库内容，也不要假装能回答任何问题。\n"
        "回答控制在 60 字以内。"
    ),
    user_template="{question}",
)


# Agent 专用注册表（与业务 PROMPT_REGISTRY、判官 JUDGE_PROMPTS 三方隔离）
AGENT_PROMPTS: dict[str, PromptTemplate] = {
    t.name: t
    for t in (QUERY_CLASSIFY, RELEVANCE_GRADE, QUERY_REWRITE, CHITCHAT)
}


def get_agent_prompt(name: str) -> PromptTemplate:
    if name not in AGENT_PROMPTS:
        raise ValueError(
            f"未知 Agent 模板 '{name}'，可用: {', '.join(sorted(AGENT_PROMPTS))}"
        )
    return AGENT_PROMPTS[name]
