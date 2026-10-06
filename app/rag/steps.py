"""
RAG 步骤层 — 可组合的原子步骤

**为什么要抽出这一层**：
  `RagPipeline.answer()` 原本是一个 90 行的单体方法，7 个步骤硬编码在一条直线里。
  Agent 改造需要把这些步骤重新编排成状态机（加分支、加循环、加工具），
  如果只能整体调用，就无法复用。

  抽成独立函数后：
    - `RagPipeline`       → 顺序调用（命令式编排）
    - `app/agent/nodes.py` → 每个步骤包一个节点（图式编排）

  这样「普通 RAG vs LangGraph」的对比就是字面意义上的**同一批步骤、两种编排方式**，
  行为差异只来自编排逻辑本身，而不是实现细节。

**设计约束**：
  - 每个步骤尽量是纯函数或只依赖注入的参数，不读全局状态
  - 行为必须与重构前的 `answer()` 逐字节一致（256 个既有测试是回归网）
"""
import logging
from dataclasses import dataclass, field

from app.config import settings
from app.llm.client import LLMClient, LLMResponse
from app.llm.prompts import PROMPT_REGISTRY, PromptTemplate
from app.rag.citations import (
    build_context,
    extract_citation_indices,
    is_refusal_lexical,
    strip_invalid_citations,
    validate_citations,
)
from app.rag.models import Citation, RetrievalConfig
from app.services.index_service import IndexHit

logger = logging.getLogger("rag_api.rag.steps")


# ═══════════════════════════════════════════════════════════════
# 步骤 1：检索
# ═══════════════════════════════════════════════════════════════

def is_index_ready(index_service) -> bool:
    """
    索引是否可用。

    独立成函数是因为「索引未就绪」和「检索无命中」是两种不同的拒答原因，
    Agent 编排时需要分别处理（前者该提示用户建索引，后者该重试或改写查询）。
    """
    return index_service is not None and index_service.ready


async def retrieve(
    index_service,
    query: str,
    config: RetrievalConfig,
) -> list[IndexHit]:
    """
    语义检索（按相似度降序）。

    不处理「索引未就绪」——调用方先用 `is_index_ready()` 判断。
    """
    return await index_service.search(
        query,
        retrieve_k=config.retrieve_k,
        min_score=config.min_score,
        final_k=config.final_k,
        document_id=config.document_id,
    )


# ═══════════════════════════════════════════════════════════════
# 步骤 2：装配上下文
# ═══════════════════════════════════════════════════════════════

def assemble_context(hits: list[IndexHit], *, max_chars: int = 6000) -> str:
    """把检索结果装配成带 [n] 编号的上下文文本"""
    return build_context(hits, max_chars=max_chars)


# ═══════════════════════════════════════════════════════════════
# 步骤 3：解析 prompt 模板
# ═══════════════════════════════════════════════════════════════

def resolve_prompt(prompt_name: str) -> PromptTemplate:
    """
    按名取 prompt 模板。

    Raises:
        ValueError: 模板名不存在（附可用列表，便于排查拼写错误）
    """
    template = PROMPT_REGISTRY.get(prompt_name)
    if template is None:
        raise ValueError(
            f"未知 prompt 模板 '{prompt_name}'，"
            f"可用: {', '.join(sorted(PROMPT_REGISTRY))}"
        )
    return template


# ═══════════════════════════════════════════════════════════════
# 步骤 4：生成
# ═══════════════════════════════════════════════════════════════

async def generate_answer(
    llm: LLMClient,
    question: str,
    context: str,
    *,
    config: RetrievalConfig,
    history: list[dict] | None = None,
    thinking_enabled: bool | None = None,
) -> LLMResponse:
    """
    渲染模板并调用 LLM 生成答案。

    Args:
        thinking_enabled: None = 用 settings.EVAL_GEN_THINKING_ENABLED
            （评测场景默认关思考降本提速；Agent 场景可显式开启）
    """
    template = resolve_prompt(config.prompt_name)
    system, user_message = template.render(context=context, input=question)

    return await llm.generate(
        system=system,
        user_message=user_message,
        history=history,
        thinking_enabled=(
            settings.EVAL_GEN_THINKING_ENABLED
            if thinking_enabled is None
            else thinking_enabled
        ),
        temperature=config.temperature,
    )


# ═══════════════════════════════════════════════════════════════
# 步骤 5：引用校验与映射
# ═══════════════════════════════════════════════════════════════

@dataclass
class CitationResult:
    """引用处理的结果"""

    text: str                                   # 剔除越界编号后的答案
    citations: list[Citation] = field(default_factory=list)
    validity: float = 1.0                       # 引用编号有效率
    invalid: list[int] = field(default_factory=list)   # 越界的编号
    is_refusal: bool = False                    # 词法层面的拒答识别


def attach_citations(answer_text: str, hits: list[IndexHit]) -> CitationResult:
    """
    校验 LLM 输出版本中的 [n] 引用编号，并把有效编号映射回检索结果。

    LLM 可能编造 [9] 这种不存在的编号 —— 必须剔掉，否则前端渲染出死链。
    这个检查是纯正则的，零成本，但能抓到判官抓不到的「编造引用」类幻觉。
    """
    raw = answer_text or ""
    validity, invalid = validate_citations(raw, len(hits))

    if invalid:
        logger.warning(
            "LLM 输出了越界引用编号 %s（有效范围 1-%d），已从答案中剔除",
            invalid, len(hits),
        )
        raw = strip_invalid_citations(raw, len(hits))

    # 只有被答案真正引用到的命中才作为引用来源返回
    cited = [i for i in extract_citation_indices(raw) if 1 <= i <= len(hits)]
    citations = [Citation.from_hit(i, hits[i - 1]) for i in cited]

    return CitationResult(
        text=raw,
        citations=citations,
        validity=validity,
        invalid=invalid,
        is_refusal=is_refusal_lexical(raw),
    )
