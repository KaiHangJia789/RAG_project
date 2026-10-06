"""
RagPipeline — 检索增强生成的完整闭环（命令式编排）

流程:
    检索 → 阈值拒答（不发 LLM）→ 装配带编号上下文 → LLM 生成
         → 解析 [n] 引用 → 校验编号有效性 → 返回 RagAnswer

**架构说明（Week11 重构）**：
  具体步骤已抽到 `app/rag/steps.py`，本类只负责**按顺序编排**它们。
  与之对应的是 `app/agent/graphs.py` —— 用 LangGraph 把同一批步骤
  编排成状态机（带条件分支与循环）。两者共用 steps.py，所以
  「普通 RAG vs LangGraph」的差异纯粹来自编排方式，而非实现细节。

设计要点:
  - **阈值层拒答不发 LLM**：检索全低于阈值时直接返回，省掉一次昂贵调用，
    响应也从数秒降到毫秒级。
  - **依赖全部构造注入**（沿用 experiments.py:47 的 `or` 范式），
    测试可替换任意一环而不碰网络。
"""
import logging
import time

from app.llm.client import LLMClient
from app.llm.observability import current_trace_id, trace_url
from app.rag import steps
from app.rag.models import RagAnswer, RefusalReason, RetrievalConfig
from app.services.index_service import IndexService

logger = logging.getLogger("rag_api.rag")


class RagPipeline:
    """RAG 问答流水线（顺序执行 steps.py 里的原子步骤）"""

    def __init__(
        self,
        index_service: IndexService | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.index_service = index_service
        self.llm = llm or LLMClient()

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

        # ── 步骤 1：索引就绪检查 ──
        if not steps.is_index_ready(self.index_service):
            base.answer = "知识库尚未就绪，请先上传文档或构建索引。"
            base.refused = True
            base.refusal_reason = RefusalReason.INDEX_NOT_READY
            base.latency_ms = (time.monotonic() - start) * 1000
            return base

        # ── 步骤 2：检索 ──
        hits = await steps.retrieve(self.index_service, question, cfg)
        base.retrieved_count = len(hits)
        base.max_score = round(hits[0].score, 4) if hits else None
        # 检索命中的文档名（按相似度降序去重）—— 评估检索质量用，与答案引用无关
        base.retrieved_docs = list(dict.fromkeys(h.filename for h in hits))

        # ── 步骤 3：阈值层拒答（不发 LLM）──
        if not hits:
            base.answer = "根据已有资料无法回答该问题。"
            base.refused = True
            base.refusal_reason = RefusalReason.BELOW_THRESHOLD
            base.latency_ms = (time.monotonic() - start) * 1000
            logger.info(
                "拒答（检索无命中）: q=%r, min_score=%s", question[:40], cfg.min_score
            )
            return base

        # ── 步骤 4：装配上下文 + 生成 ──
        context = steps.assemble_context(hits)
        resp = await steps.generate_answer(
            self.llm, question, context, config=cfg, history=history
        )
        base.llm_called = True
        base.prompt_tokens = resp.usage.prompt_tokens
        base.completion_tokens = resp.usage.completion_tokens

        # ── 步骤 5：引用校验与映射 ──
        result = steps.attach_citations(resp.text or "", hits)
        base.answer = result.text
        base.citations = result.citations
        base.citation_validity = result.validity
        base.invalid_citations = result.invalid

        # ── 步骤 6：生成层拒答识别 ──
        if result.is_refusal:
            base.refused = True
            base.refusal_reason = RefusalReason.LLM_REFUSED

        # ── 步骤 7：追踪信息 ──
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
