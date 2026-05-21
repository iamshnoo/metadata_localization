#!/usr/bin/env python3

import csv
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path("/path/to/metacul/results/final_benchmark_matrix")
ARCHIVE_BASE = Path("/path/to/metacul/results/final_benchmark_matrix_archives")


def non_ok_leaf_dirs(root: Path) -> list[tuple[Path, str]]:
    leaves = []
    for summary in root.rglob("*_eval_summary.csv"):
        try:
            with summary.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            continue
        if not rows:
            continue
        status = (rows[0].get("status") or "").strip()
        if status and status != "ok":
            leaves.append((summary.parent, status))
    return leaves


def remove_empty_dirs(root: Path) -> int:
    removed = 0
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
            removed += 1
    return removed


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = ARCHIVE_BASE / f"_archived_non_ok_{stamp}"
    archive_root.mkdir(parents=True, exist_ok=True)

    leaves = non_ok_leaf_dirs(ROOT)
    counts = Counter(status for _, status in leaves)
    moved_files = 0
    moved_dirs = 0

    manifest_rows = []
    seen_dirs = set()
    for leaf, status in leaves:
        if leaf in seen_dirs:
            continue
        seen_dirs.add(leaf)
        rel = leaf.relative_to(ROOT)
        target_dir = archive_root / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        files_here = [p for p in leaf.iterdir() if p.is_file()]
        for src in files_here:
            dst = target_dir / src.name
            shutil.move(str(src), str(dst))
            moved_files += 1
        moved_dirs += 1
        manifest_rows.append(
            {
                "status": status,
                "source_dir": str(leaf),
                "archive_dir": str(target_dir),
                "file_count": len(files_here),
            }
        )

    removed_empty = remove_empty_dirs(ROOT)

    manifest_path = archive_root / "archive_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "archive_root": str(archive_root),
                "moved_dirs": moved_dirs,
                "moved_files": moved_files,
                "removed_empty_dirs": removed_empty,
                "status_counts": counts,
                "rows": manifest_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(
        {
            "archive_root": str(archive_root),
            "moved_dirs": moved_dirs,
            "moved_files": moved_files,
            "removed_empty_dirs": removed_empty,
            "status_counts": dict(counts),
            "manifest": str(manifest_path),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
