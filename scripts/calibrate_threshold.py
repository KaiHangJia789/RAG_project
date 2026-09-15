"""
相似度阈值标定 — 从数据里取阈值，不要拍脑袋

用法:
    python scripts/calibrate_threshold.py --strategy sentence
    python scripts/calibrate_threshold.py --strategy sentence --limit 30

为什么必须标定:
    不同嵌入模型的余弦相似度分布差异极大。实测本项目的相关问题是
    0.44-0.73，无关问题约 0.30 —— 与「0.8 以上才算相关」的直觉相差甚远。
    拍一个 0.5 的阈值会导致大部分问题被误拒，然后你会花几小时怀疑检索坏了。

输出:
    可回答问题与不可回答问题的相似度分布，以及建议阈值（两条分布之间的谷底）。
"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Windows 控制台默认 GBK，输出 █/░/emoji 会 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from app.config import settings  # noqa: E402
from app.embedding.dashscope_embedding import DashscopeEmbeddingClient  # noqa: E402
from app.services.index_service import IndexService  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s | %(message)s")


def histogram(values: list[float], lo: float, hi: float, bins: int = 14) -> list[int]:
    counts = [0] * bins
    width = (hi - lo) / bins
    for v in values:
        idx = min(bins - 1, max(0, int((v - lo) / width)))
        counts[idx] += 1
    return counts


async def main() -> None:
    ap = argparse.ArgumentParser(description="相似度阈值标定")
    ap.add_argument("--strategy", default=settings.CHUNK_STRATEGY)
    ap.add_argument("--evalset", type=Path, default=settings.EVAL_DIR / "rag_eval_set.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = json.loads(args.evalset.read_text(encoding="utf-8"))["items"]
    if args.limit:
        items = items[: args.limit]

    index = IndexService(DashscopeEmbeddingClient(), strategy=args.strategy)
    index.load()
    if not index.ready:
        print(f"索引为空（策略={args.strategy}）。先跑：")
        print(f"  python scripts/build_index.py --strategy {args.strategy}")
        sys.exit(1)

    print(f"索引策略: {args.strategy}，{index.size} 条向量")
    print(f"评测集: {len(items)} 题")
    print("检索每题 top1 相似度 ...")
    print()

    pos_scores: list[float] = []
    neg_scores: list[float] = []
    pos_miss: list[str] = []

    for item in items:
        hits = await index.search(item["question"], retrieve_k=20, final_k=1)
        top1 = hits[0].score if hits else 0.0

        if item.get("answerable"):
            pos_scores.append(top1)
            # 顺带做语料质量自检：可答题却没命中标注来源 → 语料或标注有问题
            refs = {s["document"] for s in item.get("reference_sources", [])}
            if hits and hits[0].record.filename not in refs:
                pos_miss.append(item["id"])
        else:
            neg_scores.append(top1)

    def stats(name: str, vals: list[float]) -> None:
        if not vals:
            print(f"{name}: 无数据")
            return
        s = sorted(vals)
        n = len(s)
        print(
            f"{name} (n={n}): "
            f"min={s[0]:.3f}  p25={s[n//4]:.3f}  中位={s[n//2]:.3f}  "
            f"p75={s[3*n//4]:.3f}  max={s[-1]:.3f}  "
            f"均值={sum(s)/n:.3f}"
        )

    print("=" * 72)
    stats("可回答问题  ", pos_scores)
    stats("不可回答问题", neg_scores)
    print("=" * 72)
    print()

    # 分布直方图
    if pos_scores and neg_scores:
        lo = min(min(pos_scores), min(neg_scores))
        hi = max(max(pos_scores), max(neg_scores))
        if hi - lo < 1e-6:
            hi = lo + 0.1
        bins = 14
        hp = histogram(pos_scores, lo, hi, bins)
        hn = histogram(neg_scores, lo, hi, bins)
        width = (hi - lo) / bins

        print(f"{'区间':<16} {'可答':>6} {'不可答':>8}")
        print("-" * 34)
        for i in range(bins):
            left = lo + i * width
            bar_p = "█" * hp[i]
            bar_n = "░" * hn[i]
            print(f"{left:.3f}-{left+width:.3f}  {hp[i]:>3} {bar_p:<10} {hn[i]:>3} {bar_n}")
        print()

        # 建议阈值：最大化「可答通过率 - 不可答通过率」的切分点
        best_t, best_gain = 0.0, -1.0
        for i in range(bins + 1):
            t = lo + i * width
            pass_pos = sum(1 for v in pos_scores if v >= t) / len(pos_scores)
            pass_neg = sum(1 for v in neg_scores if v >= t) / len(neg_scores)
            gain = pass_pos - pass_neg
            if gain > best_gain:
                best_gain, best_t = gain, t

        print(f"📊 建议阈值: {best_t:.3f}")
        print(f"   该点可答题通过率 {sum(1 for v in pos_scores if v >= best_t)/len(pos_scores):.0%}，"
              f"不可答题通过率 {sum(1 for v in neg_scores if v >= best_t)/len(neg_scores):.0%}")
        overlap = sum(1 for v in pos_scores if v < best_t)
        if overlap:
            print(f"   ⚠️ {overlap} 道可答题会低于此阈值被拒（可接受少量误拒）")
        print()

    # 语料质量自检
    if pos_miss:
        print(f"⚠️  语料质量自检：{len(pos_miss)} 道可答题的 top1 不是标注来源文档")
        print(f"   {pos_miss[:10]}")
        print("   这多半是语料或标注的问题，不是检索代码的问题。")
        print("   建议：确认答案确实在那篇文档里，或为题目补充更精确的来源标注。")
    else:
        print("✅ 语料质量自检通过：所有可答题的 top1 都命中标注来源")

    print()
    print("把选定阈值写进 .env：")
    print(f"   RAG_MIN_SCORE={best_t:.2f}")
    print("（默认 0.0 = 不过滤，保证 demo 一定能跑通）")


if __name__ == "__main__":
    asyncio.run(main())
