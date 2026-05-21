#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


DEFAULT_INPUT_DIR = Path(
    "/path/to/metacul/results/downstream_localnewsqa_pretrained_adversarial_figure9_aligned"
)
DEFAULT_OUTPUT_CSV = Path(
    "/path/to/metacul/results/plots/plot8/adversarial_pretrained_summary.csv"
)


SERIES_LABELS = {
    ("1b", "with_metadata"): "MAPLE 1B (T+, I+)",
    ("1b", "without_metadata"): "MAPLE 1B (T-, I-)",
    ("3b", "with_metadata"): "MAPLE 3B (T+, I+)",
    ("3b", "without_metadata"): "MAPLE 3B (T-, I-)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate corrected raw-pretrained adversarial LocalNewsQA JSONLs."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def parse_rate(tag: str) -> float:
    value = tag.replace("p", ".")
    return float(value)


def find_runs(input_dir: Path) -> Iterable[Tuple[str, str, float, Path]]:
    pattern = re.compile(
        r"localnewsqa_eval_target_(with_metadata|without_metadata)_custom_([13]b)_(?:tplus|tminus)_(?:adv|urlmask)_c([0-9p]+)\.jsonl$"
    )
    for path in sorted(input_dir.rglob("*.jsonl")):
        match = pattern.match(path.name)
        if not match:
            continue
        meta_tag, size, rate_tag = match.groups()
        yield size, meta_tag, parse_rate(rate_tag), path


def summarize_jsonl(path: Path) -> Dict[str, Dict[str, float]]:
    counts = {
        "overall": {"correct": 0, "total": 0},
        "explicit": {"correct": 0, "total": 0},
        "ambiguous": {"correct": 0, "total": 0},
    }
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "is_correct" not in row:
                continue
            split = "ambiguous" if str(row.get("split_type", "")).lower() == "ambiguous" else "explicit"
            is_correct = bool(row["is_correct"])
            counts["overall"]["total"] += 1
            counts["overall"]["correct"] += int(is_correct)
            counts[split]["total"] += 1
            counts[split]["correct"] += int(is_correct)
    out = {}
    for split, stats in counts.items():
        total = stats["total"]
        out[split] = {
            "accuracy": (stats["correct"] / total) if total else 0.0,
            "correct": stats["correct"],
            "total": total,
        }
    return out


def read_mode(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            return str(row.get("metadata_corruption_mode") or "full_mismatch")
    return "full_mismatch"


def main() -> int:
    args = parse_args()
    rows: List[Dict[str, object]] = []
    for size, meta_tag, rate, path in find_runs(args.input_dir):
        series = SERIES_LABELS[(size, meta_tag)]
        metrics = summarize_jsonl(path)
        mode = read_mode(path)
        for split, split_metrics in metrics.items():
            rows.append(
                {
                    "series": series,
                    "split": split,
                    "mode": mode,
                    "corruption_rate": rate,
                    "accuracy": split_metrics["accuracy"],
                    "correct": split_metrics["correct"],
                    "total": split_metrics["total"],
                    "path": str(path),
                }
            )

    if not rows:
        print("[warn] No adversarial corrected JSONLs found.")
        return 0

    df = pd.DataFrame(rows)
    df.sort_values(["series", "split", "corruption_rate"], inplace=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"[ok] Wrote corrected adversarial summary: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
