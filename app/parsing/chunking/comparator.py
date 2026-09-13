"""
Chunk 策略对比器
对同一批文本用多种策略切分，产出指标对比表。
"""
from pydantic import BaseModel, Field

from app.parsing.chunking.base import ChunkStrategy
from app.parsing.chunking.registry import get_chunker


class ChunkStrategyMetrics(BaseModel):
    """单个策略的切分指标"""
    strategy: str
    chunk_count: int = 0
    avg_length: float = 0.0
    max_length: int = 0
    min_length: int = 0
    total_chars: int = 0

    model_config = {"json_schema_extra": {"example": {
        "strategy": "sentence", "chunk_count": 25,
        "avg_length": 380.5, "max_length": 498, "min_length": 120,
        "total_chars": 9513,
    }}}


class ComparisonReport(BaseModel):
    """切分对比报告"""
    text_count: int
    results: list[ChunkStrategyMetrics] = Field(default_factory=list)
    comparison_table: str = ""


class ChunkComparator:
    """对多篇文本跑多种切分策略，聚合指标"""

    def __init__(self, chunkers: dict[str, ChunkStrategy] | None = None):
        self.chunkers = chunkers or {
            name: get_chunker(name) for name in ["fixed_size", "paragraph", "sentence"]
        }

    def compare(self, texts: list[str]) -> ComparisonReport:
        """
        对一批文本跑所有策略，聚合指标。

        Args:
            texts: 待切分的文本列表

        Returns:
            ComparisonReport（含各策略指标 + markdown 对比表）
        """
        results: list[ChunkStrategyMetrics] = []

        for strategy_name, chunker in self.chunkers.items():
            metrics = self._compute_metrics(strategy_name, chunker, texts)
            results.append(metrics)

        table = self._build_table(results)
        return ComparisonReport(
            text_count=len(texts),
            results=results,
            comparison_table=table,
        )

    @staticmethod
    def _compute_metrics(
        strategy_name: str, chunker: ChunkStrategy, texts: list[str]
    ) -> ChunkStrategyMetrics:
        """对一批文本跑单个策略，聚合指标"""
        all_chunks = []
        for text in texts:
            all_chunks.extend(chunker.chunk(text))

        if not all_chunks:
            return ChunkStrategyMetrics(strategy=strategy_name)

        lengths = [c.length for c in all_chunks]
        return ChunkStrategyMetrics(
            strategy=strategy_name,
            chunk_count=len(all_chunks),
            avg_length=round(sum(lengths) / len(lengths), 1),
            max_length=max(lengths),
            min_length=min(lengths),
            total_chars=sum(lengths),
        )

    @staticmethod
    def _build_table(results: list[ChunkStrategyMetrics]) -> str:
        """生成 markdown 对比表"""
        lines = [
            "# Chunk 策略对比表",
            "",
            "| 策略 | Chunk 数 | 平均长度 | 最大长度 | 最小长度 | 总字符数 |",
            "|------|---------|---------|---------|---------|---------|",
        ]
        for r in results:
            lines.append(
                f"| {r.strategy} | {r.chunk_count} | {r.avg_length} "
                f"| {r.max_length} | {r.min_length} | {r.total_chars} |"
            )
        return "\n".join(lines)
