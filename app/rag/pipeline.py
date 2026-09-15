"""
RagPipeline — 检索增强生成的完整闭环

流程:
    检索 → 阈值拒答（不发 LLM）→ 装配带编号上下文 → LLM 生成
         → 解析 [n] 引用 → 校验编号有效性 → 返回 RagAnswer

设计要点:
  - **阈值层拒答不发 LLM**：检索全低于阈值时直接返回，省掉一次昂贵调用，
    响应也从数秒降到毫秒级。
  - **依赖全部构造注入**（沿用 experiments.py:47 的 `or` 范式），
    测试可替换任意一环而不碰网络。
  - **引用编号双向校验**：LLM 可能编造 [9] 这种不存在的编号，
    必须过滤掉，否则前端渲染出死链。
"""
import logging
import time

from app.config import settings
from app.llm.client import LLMClient
from app.llm.observability import current_trace_id, trace_url
from app.llm.prompts import PROMPT_REGISTRY
from app.rag.citations import (
    build_context,
    extract_citation_indices,
    is_refusal_lexical,
    strip_invalid_citations,
    validate_citations,
)
from app.rag.models import Citation, RagAnswer, RefusalReason, RetrievalConfig
from app.services.index_service import IndexService

logger = logging.getLogger("rag_api.rag")


class RagPipeline:
    """RAG 问答流水线"""

    def __init__(
        self,
        index_service: IndexService | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.index_service = index_service
        self.llm = llm or LLMClient()

    # ═══════════════════════════════════════════════════════════
    # 主入口
    # ═══════════════════════════════════════════════════════════

    async def answer(
        self,
        question: str,
        config: RetrievalConfig | None = None,
        *,
        history: list[dict] | None = None,
    ) -> RagAnswer:
        """
        回答问题。

        Args:
            question: 用户问题
            config: 检索/生成参数（None = 用默认）
            history: 多轮历史（OpenAI 格式），用于追问场景
        """
        cfg = config or RetrievalConfig()
        start = time.monotonic()

        base = RagAnswer(question=question, answer="", config=cfg)

        # ── 1. 检索 ──
        if self.index_service is None or not self.index_service.ready:
            base.answer = "知识库尚未就绪，请先上传文档或构建索引。"
            base.refused = True
            base.refusal_reason = RefusalReason.INDEX_NOT_READY
            base.latency_ms = (time.monotonic() - start) * 1000
            return base

        hits = await self.index_service.search(
            question,
            retrieve_k=cfg.retrieve_k,
            min_score=cfg.min_score,
            final_k=cfg.final_k,
            document_id=cfg.document_id,
        )
        base.retrieved_count = len(hits)
        base.max_score = round(hits[0].score, 4) if hits else None
        # 检索命中的文档名（按相似度降序去重）—— 评估检索质量用，与答案引用无关
        base.retrieved_docs = list(dict.fromkeys(h.filename for h in hits))

        if not hits:
            # ── 2. 阈值层拒答：不发 LLM ──
            base.answer = "根据已有资料无法回答该问题。"
            base.refused = True
            base.refusal_reason = RefusalReason.BELOW_THRESHOLD
            base.latency_ms = (time.monotonic() - start) * 1000
            logger.info(
                "拒答（检索无命中）: q=%r, min_score=%s", question[:40], cfg.min_score
            )
            return base

        # ── 3. 装配带编号的上下文 ──
        context = build_context(hits)

        # ── 4. 生成 ──
        template = PROMPT_REGISTRY.get(cfg.prompt_name)
        if template is None:
            raise ValueError(
                f"未知 prompt 模板 '{cfg.prompt_name}'，"
                f"可用: {', '.join(sorted(PROMPT_REGISTRY))}"
            )
        system, user_message = template.render(context=context, input=question)

        resp = await self.llm.generate(
            system=system,
            user_message=user_message,
            history=history,
            thinking_enabled=settings.EVAL_GEN_THINKING_ENABLED,
            temperature=cfg.temperature,
        )
        base.llm_called = True
        base.prompt_tokens = resp.usage.prompt_tokens
        base.completion_tokens = resp.usage.completion_tokens

        # ── 5. 引用校验 ──
        raw_answer = resp.text or ""
        validity, invalid = validate_citations(raw_answer, len(hits))
        base.citation_validity = validity
        base.invalid_citations = invalid
        if invalid:
            logger.warning(
                "LLM 输出了越界引用编号 %s（有效范围 1-%d），已从答案中剔除",
                invalid, len(hits),
            )
            raw_answer = strip_invalid_citations(raw_answer, len(hits))

        base.answer = raw_answer

        # 只有被答案真正引用到的命中才作为引用来源返回
        cited = [i for i in extract_citation_indices(raw_answer) if 1 <= i <= len(hits)]
        base.citations = [Citation.from_hit(i, hits[i - 1]) for i in cited]

        # ── 6. 生成层拒答识别 ──
        if is_refusal_lexical(raw_answer):
            base.refused = True
            base.refusal_reason = RefusalReason.LLM_REFUSED

        # ── 7. 追踪信息 ──
        tid = current_trace_id()
        if tid:
            base.trace_id = tid
            base.trace_url = trace_url(tid)

        base.latency_ms = (time.monotonic() - start) * 1000
        logger.info(
            "问答完成: q=%r, hits=%d, cited=%d, refused=%s, %.0fms",
            question[:40], len(hits), len(base.citations), base.refused, base.latency_ms,
        )
        return base
