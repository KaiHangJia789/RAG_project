"""
Prompt 实验脚本 — 一键跑 ≥5 组 Prompt 对比实验

用法:
    python scripts/run_prompt_experiments.py

产物:
    docs/week8/prompt_comparison.md（对比表 + 详细输出）
    LangFuse Dashboard 追踪记录
"""
import asyncio
import sys
from pathlib import Path

# 确保项目根目录在 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm.client import LLMClient          # noqa: E402
from app.llm.experiments import PromptExperimentRunner  # noqa: E402
from app.llm.observability import flush        # noqa: E402

# 测试输入（对齐 RAG 场景）
TEST_INPUT = (
    "什么是 RAG（检索增强生成）？它和传统的关键词搜索有什么区别？"
    "请解释它的核心流程和适用场景。"
)

# 要跑的实验组（≥5 组）
PROMPT_NAMES = [
    "baseline",
    "structured",
    "concise",
    "json_output",
    "chain_of_thought",
    "few_shot",
]


async def main() -> None:
    client = LLMClient()
    runner = PromptExperimentRunner(client)

    print("=" * 60)
    print("Prompt 实验开始")
    print(f"测试输入: {TEST_INPUT}")
    print(f"实验组: {', '.join(PROMPT_NAMES)}")
    print("=" * 60)

    report = await runner.run(TEST_INPUT, PROMPT_NAMES)

    # 输出对比表到控制台
    print("\n" + report.comparison_table)

    # 写入 markdown 文件
    out_dir = Path(__file__).resolve().parents[1] / "docs" / "week8"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prompt_comparison.md"
    out_path.write_text(report.comparison_table, encoding="utf-8")
    print(f"\n✅ 对比表已写入: {out_path}")

    # flush LangFuse
    flush()
    print("✅ LangFuse 数据已上报（若已配置 key）")


if __name__ == "__main__":
    asyncio.run(main())
