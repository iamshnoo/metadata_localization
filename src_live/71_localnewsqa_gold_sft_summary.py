#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


ROOT = Path("/path/to/metacul")
DEFAULT_SFT_ROOT = ROOT / "results/localnewsqa_gold_20260516/sft_target"
DEFAULT_PLOT_CSV = ROOT / "results/localnewsqa_gold_20260516/plots/plot_8_sft_target_split_multiseed_gold.csv"

SFT_VARIANTS = [
    ("1B", "1B T+/I+", "1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos", "sftgold_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tplus_eplus", "with_metadata"),
    ("1B", "1B T+/I-", "1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos", "sftgold_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tplus_eminus", "without_metadata"),
    ("1B", "1B T-/I+", "1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos", "sftgold_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tminus_eplus", "with_metadata"),
    ("1B", "1B T-/I-", "1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos", "sftgold_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tminus_eminus", "without_metadata"),
    ("3B", "3B T+/I+", "3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos", "sftgold_3b_best3b_chat_tplus_eplus", "with_metadata"),
    ("3B", "3B T+/I-", "3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos", "sftgold_3b_best3b_chat_tplus_eminus", "without_metadata"),
    ("3B", "3B T-/I+", "3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos", "sftgold_3b_best3b_chat_tminus_eplus", "with_metadata"),
    ("3B", "3B T-/I-", "3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos", "sftgold_3b_best3b_chat_tminus_eminus", "without_metadata"),
]

SPLIT_ORDER = {"Overall": 0, "Explicit": 1, "Ambiguous": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate gold LocalNewsQA SFT target runs.")
    parser.add_argument("--sft-root", type=Path, default=DEFAULT_SFT_ROOT)
    parser.add_argument("--seeds", default="41,42,43,44,45")
    parser.add_argument("--plot-csv", type=Path, default=DEFAULT_PLOT_CSV)
    return parser.parse_args()


def parse_seed_list(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def mean(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    return sum(vals) / len(vals)


def sample_std(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) <= 1:
        return 0.0
    avg = sum(vals) / len(vals)
    return math.sqrt(sum((v - avg) ** 2 for v in vals) / (len(vals) - 1))


def find_single(pattern: str, root: Path) -> Optional[Path]:
    matches = sorted(root.glob(pattern))
    if not matches:
        return None
    return matches[0]


def load_metrics(path: Path) -> Dict[str, float]:
    counts = {
        "Overall": {"correct": 0, "total": 0},
        "Explicit": {"correct": 0, "total": 0},
        "Ambiguous": {"correct": 0, "total": 0},
    }
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "is_correct" not in row:
                continue
            split = "Ambiguous" if str(row.get("split_type", "")).lower() == "ambiguous" else "Explicit"
            correct = int(bool(row["is_correct"]))
            counts["Overall"]["total"] += 1
            counts["Overall"]["correct"] += correct
            counts[split]["total"] += 1
            counts[split]["correct"] += correct
    metrics = {}
    for split, stats in counts.items():
        total = stats["total"]
        metrics[f"{split}_total"] = total
        metrics[f"{split}_accuracy"] = stats["correct"] / total if total else 0.0
    return metrics


def main() -> int:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    rows = []
    for family, series, root_name, run_tag_prefix, eval_meta in SFT_VARIANTS:
        split_values = {"Overall": [], "Explicit": [], "Ambiguous": []}
        totals = {"Overall": None, "Explicit": None, "Ambiguous": None}
        for seed in seeds:
            seed_dir = args.sft_root / root_name / f"seed_{seed}"
            path = find_single(
                f"localnewsqa_eval_target_{eval_meta}_custom*{run_tag_prefix}_seed{seed}_c0*.jsonl",
                seed_dir,
            )
            if path is None:
                print(f"[missing] SFT seed={seed} {series}")
                continue
            metrics = load_metrics(path)
            for split in ("Overall", "Explicit", "Ambiguous"):
                split_values[split].append(metrics[f"{split}_accuracy"])
                totals[split] = int(metrics[f"{split}_total"])
        for split in ("Explicit", "Ambiguous", "Overall"):
            values = split_values[split]
            if not values:
                continue
            rows.append(
                {
                    "family": family,
                    "series": series,
                    "split": split,
                    "accuracy": mean(values),
                    "accuracy_std": sample_std(values),
                    "seed_count": len(values),
                    "total": totals[split],
                }
            )
    rows.sort(key=lambda row: (row["family"], row["series"], SPLIT_ORDER[row["split"]]))
    args.plot_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.plot_csv, index=False)
    print(f"[ok] wrote {args.plot_csv} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
