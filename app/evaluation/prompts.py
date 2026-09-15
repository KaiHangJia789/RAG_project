"""
判官 Prompt 库

⚠️ 关键约束：**绝不能放进 app/llm/prompts.py 的 PROMPT_REGISTRY**。

原因：tests/test_prompts.py::test_all_prompts_renderable 会遍历注册表的
**每一个**模板，用固定的 {input, context, language} 三参数去渲染。
判官模板需要 {answer} / {claims} / {chunks} 这类占位符，加进去会让那个
测试直接 KeyError 失败。所以判官 prompt 独立成表。

设计原则（详见 docs/week10/README.md）：
  1. 全部要求输出 JSON —— 自然语言的评分无法程序化解析
  2. 给出具体反例 —— 光说「判断是否相关」模型会过于宽松
  3. 批量判定 —— 一次调用判定全部条目，否则调用量翻 N 倍
  4. 明确「不得遗漏、不得增加」—— 判定数对不齐是判官不稳定的头号表现
"""
from app.llm.prompts import PromptTemplate

# 换 prompt 必须同时改版本号：缓存键包含版本，历史分数不再可比
PROMPT_VERSION = "judge-v1"


# ═══════════════════════════════════════════════════════════════
# 1. Faithfulness 第一步：拆解原子断言
# ═══════════════════════════════════════════════════════════════

FAITHFULNESS_DECOMPOSE = PromptTemplate(
    name="faithfulness_decompose",
    description="把答案拆解为最小的事实性断言列表",
    system=(
        "你是一个严谨的事实核查助手，任务是把一段「答案」拆解成最小的事实性断言。\n"
        "\n"
        "规则：\n"
        "1. 每条断言只包含一个可独立判断真伪的事实。\n"
        "2. 保留答案中的具体数字、名称、条件（如「在 X 情况下」），不要泛化。\n"
        "3. 忽略过渡语、礼貌语、格式标记；像 [1][2] 这样的引用标记不构成断言。\n"
        "4. 用答案中的原话表述，不要改写，不要补充答案里没有的信息。\n"
        "5. 如果答案是拒答（例如「根据提供的资料无法回答」），"
        "则 claims 返回空数组且 is_refusal 为 true。\n"
        "\n"
        "只输出 json，不要任何解释，不要 markdown 代码块。\n"
        '输出格式：{"claims": ["断言1", "断言2"], "is_refusal": false}'
    ),
    user_template="答案：\n{answer}",
)


# ═══════════════════════════════════════════════════════════════
# 2. Faithfulness 第二步：批量核验断言
# ═══════════════════════════════════════════════════════════════

FAITHFULNESS_VERIFY = PromptTemplate(
    name="faithfulness_verify",
    description="批量判断每条断言能否由上下文支持",
    system=(
        "你是一个严谨的事实核查助手。你会拿到一批「待核查断言」和一段「参考上下文」，"
        "请逐条判断该断言能否由参考上下文支持。\n"
        "\n"
        "判定标准（从严）：\n"
        "- supported = true：上下文中存在明确语句，或可由上下文直接推出等价表述。\n"
        "- supported = false：上下文中没有该信息；或上下文信息与该断言矛盾；"
        "或需要上下文之外的知识才能成立。\n"
        "- 数字、日期、名称不一致 → false。\n"
        "- 上下文只是「话题相关」但没有给出该具体结论 → false。"
        "不要因为话题相近就判 true。\n"
        "- 断言比上下文更绝对（上下文说「通常」，断言说「总是」）→ false。\n"
        "\n"
        "必须对每条断言给出一个判定，不得遗漏、不得增加。\n"
        "\n"
        "只输出 json，不要任何解释，不要 markdown 代码块。\n"
        '输出格式：{"verdicts": [{"id": 1, "supported": true, '
        '"evidence": "支持它的原句（不超过50字），不支持时填空字符串"}]}'
    ),
    user_template="参考上下文：\n{context}\n\n待核查断言：\n{claims}",
)


# ═══════════════════════════════════════════════════════════════
# 3. Answer Relevance：从答案反推问题
# ═══════════════════════════════════════════════════════════════

RELEVANCE_REVERSE_QUESTIONS = PromptTemplate(
    name="relevance_reverse_questions",
    description="根据答案反推它能够回答的问题（用于计算答案相关性）",
    system=(
        "你是一个问题生成助手。给定一个「答案」，请生成 3 个该答案能够回答的不同问题。\n"
        "\n"
        "要求：\n"
        "1. 每个问题都必须能被该答案直接回答。\n"
        "2. 3 个问题从不同角度切入（例如：是什么 / 为什么 / 怎么做）。\n"
        "3. 问题要具体，包含答案里的关键名词；不要用「它」「这个」这类指代。\n"
        "4. 不要照抄答案，不要回答问题本身。\n"
        "\n"
        "只输出 json，不要任何解释，不要 markdown 代码块。\n"
        '输出格式：{"questions": ["问题1", "问题2", "问题3"]}'
    ),
    user_template="答案：\n{answer}",
)


# ═══════════════════════════════════════════════════════════════
# 4. Context Precision：批量判定检索片段相关性
# ═══════════════════════════════════════════════════════════════

CONTEXT_RELEVANCE = PromptTemplate(
    name="context_relevance_judge",
    description="逐条判断检索到的上下文片段对回答问题是否有用",
    system=(
        "你是一个检索质量评估助手。给定一个「问题」、一段「参考标准答案」和若干条"
        "「检索到的上下文片段」，请逐条判断该片段对回答该问题是否有用。\n"
        "\n"
        "判定标准：\n"
        "- relevant = true：片段包含回答问题所需的信息"
        "（即使不完整、即使只是部分相关）。\n"
        "- relevant = false：片段与该问题无关；或只是重复了问题本身没有提供任何信息。\n"
        "- 判断依据是「这段文字是否有助于回答问题」，而不是「是否包含问题的字面关键词」。\n"
        "- 参考标准答案只用于帮助你理解问题的真正意图；"
        "不要因为片段与标准答案字面不同就判为不相关。\n"
        "- 片段即使正确、即使质量很高，只要与本题无关就是 false。\n"
        "\n"
        "必须对每个片段给出一个判定，不得遗漏、不得增加。\n"
        "\n"
        "只输出 json，不要任何解释，不要 markdown 代码块。\n"
        '输出格式：{"judgments": [{"id": 1, "relevant": true, "reason": "不超过20字"}]}'
    ),
    user_template=(
        "问题：\n{question}\n\n"
        "参考标准答案（仅用于理解问题意图）：\n{reference_answer}\n\n"
        "检索到的上下文片段：\n{chunks}"
    ),
)


# ═══════════════════════════════════════════════════════════════
# 5. Answer Relevance 交叉验证：判官直接打分
# ═══════════════════════════════════════════════════════════════

RELEVANCE_DIRECT = PromptTemplate(
    name="relevance_direct_score",
    description="判官直接给答案相关性打分（用于与反推问题法交叉验证）",
    system=(
        "你是一个答案质量评估助手。给定一个「问题」和一段「答案」，"
        "请评估该答案对问题的相关程度。\n"
        "\n"
        "评分标准（只看切题程度，**不看答案是否正确**）：\n"
        "- 1.0：完全切题，直接且完整地回应了问题。\n"
        "- 0.7：基本切题，但遗漏了问题的一部分，或包含少量无关内容。\n"
        "- 0.4：部分相关，只回应了问题的某个侧面，或答非所问。\n"
        "- 0.0：完全不切题，答的是另一个问题。\n"
        "\n"
        "注意：\n"
        "- 如果答案是拒答（例如「无法回答」），且该问题确实不该被回答，"
        "此时相关性按 0.0 计（因为拒答没有回应问题本身）。\n"
        "- 不要因为答案内容错误就扣分 —— 这个指标只衡量相关性。\n"
        "\n"
        "只输出 json，不要任何解释，不要 markdown 代码块。\n"
        '输出格式：{"score": 0.7, "reason": "不超过30字"}'
    ),
    user_template="问题：\n{question}\n\n答案：\n{answer}",
)


# 判官专用注册表（与业务 PROMPT_REGISTRY 隔离）
JUDGE_PROMPTS: dict[str, PromptTemplate] = {
    t.name: t
    for t in (
        FAITHFULNESS_DECOMPOSE,
        FAITHFULNESS_VERIFY,
        RELEVANCE_REVERSE_QUESTIONS,
        CONTEXT_RELEVANCE,
        RELEVANCE_DIRECT,
    )
}


def get_judge_prompt(name: str) -> PromptTemplate:
    if name not in JUDGE_PROMPTS:
        raise ValueError(
            f"未知判官模板 '{name}'，可用: {', '.join(sorted(JUDGE_PROMPTS))}"
        )
    return JUDGE_PROMPTS[name]
