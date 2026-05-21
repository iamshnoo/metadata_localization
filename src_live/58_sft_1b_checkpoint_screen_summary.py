#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import pandas as pd


def load_metrics(path: Path) -> dict:
    counts = {
        "overall": {"correct": 0, "total": 0},
        "explicit": {"correct": 0, "total": 0},
        "ambiguous": {"correct": 0, "total": 0},
    }
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            ok = int(bool(row.get("is_correct")))
            split = "ambiguous" if str(row.get("split_type", "")).lower() == "ambiguous" else "explicit"
            counts["overall"]["total"] += 1
            counts["overall"]["correct"] += ok
            counts[split]["total"] += 1
            counts[split]["correct"] += ok
    out = {}
    for split, stats in counts.items():
        out[split] = stats["correct"] / stats["total"] if stats["total"] else 0.0
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize 1B SFT checkpoint screen runs.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/path/to/metacul/results/downstream_localnewsqa_sft_1b_checkpoint_screen"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/path/to/metacul/results/analysis/sft_1b_checkpoint_screen_summary.csv"),
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=Path("/path/to/metacul/results/analysis/sft_1b_checkpoint_screen_comparison.csv"),
    )
    args = parser.parse_args()

    rows = []
    for fp in sorted(args.root.glob("checkpoint_*/*.jsonl")):
        match = re.search(r"checkpoint_(\d+)", str(fp))
        if not match:
            continue
        ckpt = int(match.group(1))
        name = fp.name
        if "tplus_eplus" in name:
            setting = "T+/I+"
        elif "tminus_eminus" in name:
            setting = "T-/I-"
        else:
            continue
        metrics = load_metrics(fp)
        rows.append(
            {
                "checkpoint": ckpt,
                "setting": setting,
                "overall": metrics["overall"],
                "explicit": metrics["explicit"],
                "ambiguous": metrics["ambiguous"],
            }
        )

    if not rows:
        print("[warn] no completed checkpoint-screen rows found")
        return 0

    df = pd.DataFrame(rows).sort_values(["checkpoint", "setting"]).reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"[ok] wrote {args.output_csv}")
    print(df.to_string(index=False))

    pivot = (
        df.pivot(index="checkpoint", columns="setting", values=["overall", "explicit", "ambiguous"])
        .sort_index()
    )
    pivot.columns = [f"{metric}_{setting}" for metric, setting in pivot.columns]
    pivot = pivot.reset_index()
    for split in ("overall", "explicit", "ambiguous"):
        plus_col = f"{split}_T+/I+"
        minus_col = f"{split}_T-/I-"
        if plus_col in pivot.columns and minus_col in pivot.columns:
            pivot[f"{split}_delta_pp"] = 100.0 * (pivot[plus_col] - pivot[minus_col])
    args.comparison_csv.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(args.comparison_csv, index=False)
    print(f"[ok] wrote {args.comparison_csv}")
    print(pivot.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
