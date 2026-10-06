"""
A/B 对比报告 — 基础 RAG vs Agentic RAG

用法:
    # 1. 先跑两组评测
    python scripts/run_rag_eval.py --mode basic --week week13 --name ab_basic
    python scripts/run_rag_eval.py --mode agent --week week13 --name ab_agent

    # 2. 生成对比报告
    python scripts/run_ab_eval.py --a ab_basic --b ab_agent

产物:
    docs/week13/ab_report.md —— 指标对比表 + 失败案例清单

**对比的公平性说明（必须写进报告）**：
  Agent 模式每题会多调用若干次 LLM（分类/评分/重写/工具轮），
  所以延迟与 token 数**天然**更高。直接比绝对值意义不大，
  有意义的是「同等质量下的成本」以及各指标本身的差异。
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import settings  # noqa: E402
from app.evaluation.evaluator import QuestionMetrics, summarize  # noqa: E402

logging.basicConfig(level=logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]

# 对比的指标与「越大越好」的方向
METRICS = [
    ("faithfulness", "Faithfulness", True),
    ("answer_relevance", "Answer Relevance", True),
    ("context_precision", "Context Precision（判官）", True),
    ("context_precision_source", "Context Precision（来源）", True),
    ("hit_rate", "Hit Rate", True),
    ("mrr", "MRR", True),
    ("citation_validity", "Citation Validity", True),
    ("hallucination_rate", "幻觉率", False),
    ("refusal_accuracy", "拒答正确率", True),
    ("false_refusal_rate", "误拒率", False),
    ("avg_latency_ms", "平均延迟(ms)", False),
]


def load_results(name: str) -> list[QuestionMetrics] | None:
    """从 data/eval/result_{name}.jsonl 读取逐题结果"""
    path = settings.EVAL_DIR / f"result_{name}.jsonl"
    if not path.exists():
        return None
    items = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            items.append(QuestionMetrics(**json.loads(line)))
        except Exception:
            continue
    return items or None


def fmt(v: float) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def delta_mark(a: float, b: float, higher_better: bool) -> str:
    """标出 b 相对 a 的变化方向"""
    diff = b - a
    if abs(diff) < 1e-4:
        return "  —"
    better = (diff > 0) if higher_better else (diff < 0)
    return f"{'+' if diff > 0 else ''}{diff:.4f} {'✅' if better else '⚠️'}"


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B 对比报告")
    ap.add_argument("--a", default="ab_basic", help="A 组配置名（基础 RAG）")
    ap.add_argument("--b", default="ab_agent", help="B 组配置名（Agentic RAG）")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/week13/ab_report.md")
    args = ap.parse_args()

    ra, rb = load_results(args.a), load_results(args.b)
    if ra is None or rb is None:
        missing = args.a if ra is None else args.b
        print(f"缺少结果文件: data/eval/result_{missing}.jsonl")
        print("请先运行:")
        print(f"  python scripts/run_rag_eval.py --mode basic --week week13 --name {args.a}")
        print(f"  python scripts/run_rag_eval.py --mode agent --week week13 --name {args.b}")
        sys.exit(1)

    A = summarize(ra, config=args.a)
    B = summarize(rb, config=args.b)

    lines = [
        "# 基础 RAG vs Agentic RAG — A/B 对比报告",
        "",
        f"- **A 组**（基线）：`{args.a}` — {A.total} 题",
        f"- **B 组**（实验）：`{args.b}` — {B.total} 题",
        "- 同一评测集、同一索引、同一判官缓存",
        "",
        "> **公平性说明**：Agent 模式每题会多调用若干次 LLM（查询分类、相关性评分、"
        "查询重写、工具轮），所以延迟与 token 天然更高。**直接比绝对值意义不大**，"
        "有意义的是各质量指标的差异，以及「同等质量下的额外成本」。",
        "",
        "## 一、指标对比",
        "",
        "| 指标 | A: 基础 RAG | B: Agentic | 差值 | 方向 |",
        "|---|---|---|---|---|",
    ]

    for attr, label, higher_better in METRICS:
        va, vb = getattr(A, attr, 0.0), getattr(B, attr, 0.0)
        lines.append(
            f"| {label} | {fmt(va)} | {fmt(vb)} | {delta_mark(va, vb, higher_better)} | |"
        )

    # 收集失败案例
    lines += ["", "## 二、逐题差异", ""]
    map_a = {m.question_id: m for m in ra}
    map_b = {m.question_id: m for m in rb}
    common = sorted(set(map_a) & set(map_b))

    diffs = []
    for qid in common:
        ma, mb = map_a[qid], map_b[qid]
        fa = ma.faithfulness if ma.faithfulness is not None else 0.0
        fb = mb.faithfulness if mb.faithfulness is not None else 0.0
        if abs(fa - fb) > 0.01 or ma.refused != mb.refused:
            diffs.append((qid, ma, mb, fb - fa))

    diffs.sort(key=lambda x: -abs(x[3]))

    if not diffs:
        lines.append("两组在全部题目上表现一致。")
    else:
        lines += [
            f"共 {len(diffs)} 题存在差异（按 Faithfulness 差值排序，最多列 15 条）：",
            "",
            "| 题号 | 问题 | A 忠实度 | B 忠实度 | A 拒答 | B 拒答 |",
            "|---|---|---|---|---|---|",
        ]
        for qid, ma, mb, _ in diffs[:15]:
            q = (ma.question or "")[:36].replace("|", "／")
            fa = f"{ma.faithfulness:.2f}" if ma.faithfulness is not None else "—"
            fb = f"{mb.faithfulness:.2f}" if mb.faithfulness is not None else "—"
            lines.append(
                f"| {qid} | {q} | {fa} | {fb} | "
                f"{'是' if ma.refused else '否'} | {'是' if mb.refused else '否'} |"
            )

    # 成本对比
    lines += [
        "",
        "## 三、成本与效率",
        "",
        "| 项目 | A: 基础 RAG | B: Agentic |",
        "|---|---|---|",
        f"| 平均延迟 | {A.avg_latency_ms:.0f} ms | {B.avg_latency_ms:.0f} ms |",
        f"| LLM 调用题数 | {A.llm_calls} | {B.llm_calls} |",
        "",
        "## 四、结论要点",
        "",
        "（运行后按实际数据补充，下面为分析框架）",
        "",
        "1. **质量差异**：看 Faithfulness / Answer Relevance 是否显著变化。",
        "2. **检索质量**：看 Hit Rate / MRR 是否变化（这两个是零成本确定性指标，最可信）。",
        "3. **拒答行为**：看误拒率是否上升（Agent 的额外判断层可能过度拒答）。",
        "4. **成本代价**：Agent 的延迟倍率是多少？这个代价换来的是否值得？",
        "",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")

    print(f"A/B 报告已生成: {args.out.relative_to(ROOT)}")
    print()
    print(f"  A({args.a}): Faithfulness={A.faithfulness:.4f} "
          f"HitRate={A.hit_rate:.4f} 延迟={A.avg_latency_ms:.0f}ms 误拒率={A.false_refusal_rate:.4f}")
    print(f"  B({args.b}): Faithfulness={B.faithfulness:.4f} "
          f"HitRate={B.hit_rate:.4f} 延迟={B.avg_latency_ms:.0f}ms 误拒率={B.false_refusal_rate:.4f}")


if __name__ == "__main__":
    main()
