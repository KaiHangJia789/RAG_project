"""
清理 uploads/ 下的孤儿文件（磁盘上有、但 DB 里没有任何记录引用）。

用法:
    python scripts/clean_orphan_files.py            # dry-run，只列出不删
    python scripts/clean_orphan_files.py --delete   # 真正删除

安全护栏（任一不满足就中止，绝不误删）:
  1. DB 查询必须成功 —— 查询失败时 known 集合为空，会把所有文件误判为孤儿
  2. DB 里必须至少有 1 条 documents 记录 —— 0 条说明连错库了
  3. 只处理 settings.UPLOAD_DIR 目录内的文件
  4. 绝不删除被 DB 引用的路径
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.db.connection import Database  # noqa: E402


def _norm(path: str) -> str:
    """路径归一化：统一分隔符，便于跨平台比较"""
    return str(path).replace("\\", "/")


async def collect_orphans() -> tuple[list[Path], list[Path]]:
    """
    返回 (orphans, all_files)。

    Raises:
        RuntimeError: 触发安全护栏
    """
    db = Database()
    try:
        await db.connect(settings.DATABASE_DSN)
        rows = await db.fetch_all("SELECT id, storage_path FROM documents")
    except Exception as e:
        raise RuntimeError(
            f"数据库查询失败，已中止以免误删全部文件: {e}"
        ) from e
    finally:
        try:
            await db.disconnect()
        except Exception:
            pass

    # 护栏 2：DB 为空说明连错库了
    if not rows:
        raise RuntimeError(
            "documents 表 0 条记录，疑似连错库，已中止（否则会把所有文件当孤儿删掉）"
        )

    known = {_norm(r["storage_path"]) for r in rows}
    print(f"DB 记录 {len(rows)} 条，引用 {len(known)} 个路径：")
    for r in rows:
        full = settings.UPLOAD_DIR.parent / r["storage_path"]
        print(f"   {r['storage_path']:<46} 磁盘上{'存在' if full.exists() else '不存在'}")
    print()

    upload_root = settings.UPLOAD_DIR.resolve()
    all_files = sorted(
        p for p in settings.UPLOAD_DIR.rglob("*") if p.is_file()
    )

    orphans = []
    for p in all_files:
        # 护栏 3：只在 uploads/ 内操作
        if upload_root not in p.resolve().parents:
            continue
        # 护栏 4：DB 引用的一律跳过
        if _norm(p) in known:
            continue
        orphans.append(p)

    return orphans, all_files


def prune_empty_dirs() -> int:
    """删除 uploads/ 下的空目录（保留 uploads 根目录）"""
    removed = 0
    root = settings.UPLOAD_DIR.resolve()
    # 自底向上，先删深层的
    for d in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if d.is_dir() and d.resolve() != root and not any(d.iterdir()):
            d.rmdir()
            removed += 1
    return removed


async def main() -> None:
    parser = argparse.ArgumentParser(description="清理 uploads/ 下的孤儿文件")
    parser.add_argument("--delete", action="store_true", help="真正执行删除")
    args = parser.parse_args()

    try:
        orphans, all_files = await collect_orphans()
    except RuntimeError as e:
        print(f"[中止] {e}")
        sys.exit(1)

    total_bytes = sum(p.stat().st_size for p in orphans)
    print(f"磁盘文件 {len(all_files)} 个，其中孤儿 {len(orphans)} 个：")
    for p in orphans:
        print(f"   {p}  ({p.stat().st_size} B)")
    print()
    print(f"合计 {len(orphans)} 个，{total_bytes / 1024 / 1024:.2f} MB")

    if not args.delete:
        print()
        print("--- DRY RUN，未删除任何文件。确认无误后加 --delete 执行 ---")
        return

    print()
    deleted = 0
    failures: list[tuple[Path, str]] = []
    for p in orphans:
        try:
            p.unlink()
            deleted += 1
        except OSError as e:
            failures.append((p, str(e)))

    removed_dirs = prune_empty_dirs()

    print(f"已删除 {deleted} 个文件，释放 {total_bytes / 1024 / 1024:.2f} MB")
    print(f"已清理 {removed_dirs} 个空目录")
    if failures:
        print(f"删除失败 {len(failures)} 个：")
        for p, err in failures:
            print(f"   {p}: {err}")

    # 复核
    remaining = [p for p in settings.UPLOAD_DIR.rglob("*") if p.is_file()]
    print(f"复核：uploads/ 下剩余 {len(remaining)} 个文件")


if __name__ == "__main__":
    asyncio.run(main())
