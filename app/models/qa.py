"""问答接口的请求/响应模型"""
from pydantic import BaseModel, Field


class QARequest(BaseModel):
    """问答请求"""

    question: str = Field(
        ..., min_length=1, max_length=2000,
        description="用户问题", examples=["什么是向量检索？"],
    )
    strategy: str = Field(
        default="splitter",
        description="使用的 chunk 策略索引（对比实验用）",
    )
    retrieve_k: int = Field(
        default=20, ge=1, le=200,
        description="FAISS 单次取回条数（阈值过滤前的候选集大小）",
    )
    final_k: int = Field(
        default=5, ge=1, le=50,
        description="最终喂给 LLM 的上下文条数",
    )
    min_score: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description="相似度阈值，低于此值的检索结果被丢弃；0 = 不过滤",
    )
    prompt_name: str = Field(
        default="rag_context_cited",
        description="使用的 prompt 模板名",
    )
    document_id: str | None = Field(
        default=None,
        description="限定在某文档内检索（None = 全库）",
    )
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0,
        description="生成温度；None = 用模型默认",
    )


class ChunkItem(BaseModel):
    """chunk 上下文中的单项"""

    chunk_id: str
    chunk_index: int
    text: str
    page_number: int | None = None
    is_target: bool = False        # 是否为被引用的那一块（前端据此滚动定位）


class ChunkContext(BaseModel):
    """chunk 及其前后文（引用可定位的返回体）"""

    document_id: str
    filename: str = ""
    chunk_strategy: str = "splitter"
    items: list[ChunkItem] = Field(default_factory=list)


class QAStats(BaseModel):
    """索引概况"""

    strategy: str
    dim: int
    vectors: int
    records: int
    documents: int
    pages: int
    index_path: str
    exists: bool
