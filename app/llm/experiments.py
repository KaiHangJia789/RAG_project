"""
Prompt 实验编排器

对同一输入跑多组 Prompt，产出对比表。这是"Prompt 实验对比表（≥5组）"
验收产物的生成器。
"""
import json
import logging
from pydantic import BaseModel, Field

from app.llm.client import LLMClient, LLMUsage
from app.llm.observability import observe
from app.llm.prompts import PROMPT_REGISTRY

logger = logging.getLogger("rag_api.llm.experiments")


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

class ExperimentResult(BaseModel):
    """单组实验结果"""
    prompt_name: str
    prompt_system: str
    output_text: str
    latency_ms: float
    usage: LLMUsage
    reasoning_content: str | None = None
    metrics: dict = Field(default_factory=dict)


class ExperimentReport(BaseModel):
    """完整实验报告"""
    input_text: str
    results: list[ExperimentResult]
    comparison_table: str                       # Markdown 对比表


# ═══════════════════════════════════════════════════════════════
# 实验编排器
# ═══════════════════════════════════════════════════════════════

class PromptExperimentRunner:
    """依次跑各组 prompt，收集结果 + 自动指标"""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    @observe()                                   # 创建 trace，把各组 generation 归到其下
    async def run(
        self,
        input_text: str,
        prompt_names: list[str] | None = None,
    ) -> ExperimentReport:
        """
        依次跑各组 prompt。

        Args:
            input_text: 测试输入
            prompt_names: 要跑的 prompt 名（默认跑全部注册表）
        """
        names = prompt_names or list(PROMPT_REGISTRY.keys())
        results: list[ExperimentResult] = []

        for name in names:
            template = PROMPT_REGISTRY.get(name)
            if template is None:
                logger.warning("未知 prompt 名: %s，跳过", name)
                continue

            system, user_message = template.render(input=input_text)
            logger.info("跑实验 [%s]: %s", name, template.description)

            resp = await self.llm.generate(
                system=system,
                user_message=user_message,
            )

            result = ExperimentResult(
                prompt_name=name,
                prompt_system=system,
                output_text=resp.text,
                latency_ms=resp.latency_ms,
                usage=resp.usage,
                reasoning_content=resp.reasoning_content,
            )
            result.metrics = self._compute_metrics(result)
            results.append(result)

        comparison_table = self._build_comparison_table(input_text, results)
        return ExperimentReport(
            input_text=input_text,
            results=results,
            comparison_table=comparison_table,
        )

    # ═══════════════════════════════════════════════════════════════
    # 自动指标
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _compute_metrics(result: ExperimentResult) -> dict:
        """计算自动指标"""
        text = result.output_text
        return {
            "char_count": len(text),                    # 字符数
            "word_count": len(text.split()),            # 词数（中文按空格近似）
            "has_json": _is_parseable_json(text),       # 是否可解析 JSON
            "has_citation": _has_citation(text),        # 是否含引用/来源标记
            "token_total": result.usage.total_tokens,
        }

    # ═══════════════════════════════════════════════════════════════
    # 对比表生成
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _build_comparison_table(
        input_text: str, results: list[ExperimentResult]
    ) -> str:
        """生成 Markdown 对比表"""
        lines = [
            "# Prompt 实验对比表",
            "",
            f"**测试输入**: {input_text}",
            "",
            "| Prompt | 输出长度(字符) | 延迟(ms) | 总 Token | 含 JSON | 含引用 |",
            "|--------|----------------|----------|---------|---------|--------|",
        ]
        for r in results:
            m = r.metrics
            lines.append(
                f"| {r.prompt_name} | {m['char_count']} | {r.latency_ms:.0f} "
                f"| {m['token_total']} | {'✅' if m['has_json'] else '—'} "
                f"| {'✅' if m['has_citation'] else '—'} |"
            )
        lines.append("")
        lines.append("## 详细输出")
        lines.append("")
        for r in results:
            lines.append(f"### {r.prompt_name}")
            lines.append("")
            lines.append("```")
            lines.append(r.output_text[:500])
            if len(r.output_text) > 500:
                lines.append("...（截断）")
            lines.append("```")
            lines.append("")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def _is_parseable_json(text: str) -> bool:
    """判断输出是否是可解析的 JSON"""
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        # 尝试提取 JSON 子串
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return False
        return False


def _has_citation(text: str) -> bool:
    """判断输出是否含引用/来源标记（如 [1]、来源：、reference 等）"""
    markers = ["[1]", "[2]", "来源", "reference", "Reference", "引用", "参见"]
    return any(m in text for m in markers)
