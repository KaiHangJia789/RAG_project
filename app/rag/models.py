"""RAG 问答链路的领域模型"""
from enum import Enum

from pydantic import BaseModel, Field

from app.services.index_service import IndexHit


class RefusalReason(str, Enum):
    """拒答原因"""

    NO_HIT = "no_hit"                         # 索引为空或检索无结果
    BELOW_THRESHOLD = "below_threshold"       # 有结果但全部低于相似度阈值
    LLM_REFUSED = "llm_refused"               # 检索到了，但 LLM 判断无法回答
    INDEX_NOT_READY = "index_not_ready"       # 索引未构建


class RetrievalConfig(BaseModel):
    """一次检索+生成的参数组合（对比实验的配置维度）"""

    strategy: str = "splitter"
    retrieve_k: int = Field(default=20, ge=1, le=200)
    min_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    final_k: int = Field(default=5, ge=1, le=50)
    prompt_name: str = "rag_context_cited"
    document_id: str | None = None
    temperature: float | None = None

    def describe(self) -> str:
        """人类可读的配置摘要（写进报告与 LangFuse run 名）"""
        parts = [f"strategy={self.strategy}", f"final_k={self.final_k}"]
        if self.min_score > 0:
            parts.append(f"min_score={self.min_score}")
        if self.prompt_name != "rag_context_cited":
            parts.append(f"prompt={self.prompt_name}")
        return ", ".join(parts)

    def as_dict(self) -> dict:
        return self.model_dump()


class Citation(BaseModel):
    """引用来源（前端"可点击/可定位"的数据来源）"""

    index: int                                  # 答案里的 [n] 编号
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    score: float
    text: str                                   # 完整 chunk 文本（点击展开）
    page_number: int | None = None

    @classmethod
    def from_hit(cls, index: int, hit: IndexHit) -> "Citation":
        return cls(
            index=index,
            chunk_id=hit.record.chunk_id,
            document_id=hit.record.document_id,
            filename=hit.record.filename,
            chunk_index=hit.record.chunk_index,
            score=round(hit.score, 4),
            text=hit.record.text,
            page_number=hit.record.page_number,
        )


class RagAnswer(BaseModel):
    """一次问答的完整结果"""

    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: RefusalReason | None = None

    # ── 可观测性 ──
    config: RetrievalConfig | None = None
    retrieved_count: int = 0                    # 检索命中数（过滤前）
    # 检索命中的文档名（按相似度降序，去重）。**不是**答案引用的文档 ——
    # 评估检索质量（Hit Rate / MRR）必须用这个，用 citations 会混入
    # 「LLM 是否引用」的变量，污染纯检索指标。
    retrieved_docs: list[str] = Field(default_factory=list)
    max_score: float | None = None              # 最高相似度（调阈值的关键信号）
    citation_validity: float = 1.0              # 引用编号有效率
    invalid_citations: list[int] = Field(default_factory=list)
    latency_ms: float = 0.0
    llm_called: bool = False                    # 是否真的调了 LLM（拒答时为 False）
    prompt_tokens: int = 0
    completion_tokens: int = 0
    trace_id: str | None = None
    trace_url: str | None = None
