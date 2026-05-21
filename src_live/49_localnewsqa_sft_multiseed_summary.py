#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


DEFAULT_SFT_ROOT = Path(
    "/path/to/metacul/results/downstream_localnewsqa_sft_figure9_full_multiseed"
)
DEFAULT_PLOT_CSV = Path(
    "/path/to/metacul/results/plots/plot8/plot_8_sft_target_split_multiseed.csv"
)

SFT_VARIANTS = [
    {
        "family": "1B",
        "series": "1B T+/I+",
        "root_name": "1b_chat_codeg_labels_question_final",
        "run_tag": "1b_chat_tplus_eplus",
        "eval_meta_tag": "with_metadata",
    },
    {
        "family": "1B",
        "series": "1B T+/I-",
        "root_name": "1b_chat_codeg_labels_question_final",
        "run_tag": "1b_chat_tplus_eminus",
        "eval_meta_tag": "without_metadata",
    },
    {
        "family": "1B",
        "series": "1B T-/I+",
        "root_name": "1b_chat_codeg_labels_question_final",
        "run_tag": "1b_chat_tminus_eplus",
        "eval_meta_tag": "with_metadata",
    },
    {
        "family": "1B",
        "series": "1B T-/I-",
        "root_name": "1b_chat_codeg_labels_question_final",
        "run_tag": "1b_chat_tminus_eminus",
        "eval_meta_tag": "without_metadata",
    },
    {
        "family": "3B",
        "series": "3B T+/I+",
        "root_name": "3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos",
        "run_tag": "3b_best3b_chat_tplus_eplus",
        "eval_meta_tag": "with_metadata",
    },
    {
        "family": "3B",
        "series": "3B T+/I-",
        "root_name": "3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos",
        "run_tag": "3b_best3b_chat_tplus_eminus",
        "eval_meta_tag": "without_metadata",
    },
    {
        "family": "3B",
        "series": "3B T-/I+",
        "root_name": "3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos",
        "run_tag": "3b_best3b_chat_tminus_eplus",
        "eval_meta_tag": "with_metadata",
    },
    {
        "family": "3B",
        "series": "3B T-/I-",
        "root_name": "3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos",
        "run_tag": "3b_best3b_chat_tminus_eminus",
        "eval_meta_tag": "without_metadata",
    },
]

SPLIT_ORDER = {"Overall": 0, "Explicit": 1, "Ambiguous": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate SFT LocalNewsQA shuffled option-text results across multiple seeds."
    )
    parser.add_argument(
        "--sft-root",
        type=Path,
        default=DEFAULT_SFT_ROOT,
        help="Root with per-seed SFT LocalNewsQA runs for the appendix figure.",
    )
    parser.add_argument(
        "--seeds",
        default="41,42,43,44,45",
        help="Comma-separated seed list used for option shuffling.",
    )
    parser.add_argument("--plot-csv", type=Path, default=DEFAULT_PLOT_CSV)
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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


def load_localnewsqa_metrics(path: Path) -> Dict[str, float]:
    counts = {
        "Overall": {"correct": 0, "total": 0},
        "Explicit": {"correct": 0, "total": 0},
        "Ambiguous": {"correct": 0, "total": 0},
    }
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "is_correct" not in row:
                continue
            split_type = str(row.get("split_type", "")).strip().lower()
            split_name = "Ambiguous" if split_type == "ambiguous" else "Explicit"
            is_correct = bool(row["is_correct"])
            counts["Overall"]["total"] += 1
            counts["Overall"]["correct"] += int(is_correct)
            counts[split_name]["total"] += 1
            counts[split_name]["correct"] += int(is_correct)

    metrics: Dict[str, float] = {}
    for split, stats in counts.items():
        total = stats["total"]
        metrics[f"{split}_total"] = total
        metrics[f"{split}_correct"] = stats["correct"]
        metrics[f"{split}_accuracy"] = stats["correct"] / total if total else 0.0
    return metrics


def aggregate_sft(args: argparse.Namespace, seeds: List[int]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for variant in SFT_VARIANTS:
        per_split_values: Dict[str, List[float]] = {
            "Overall": [],
            "Explicit": [],
            "Ambiguous": [],
        }
        total_rows = {"Overall": None, "Explicit": None, "Ambiguous": None}
        for seed in seeds:
            seed_dir = args.sft_root / variant["root_name"] / f"seed_{seed}"
            path = find_single(
                f"localnewsqa_eval_target_{variant['eval_meta_tag']}_custom*{variant['run_tag']}*_c0*.jsonl",
                seed_dir,
            )
            if path is None:
                print(f"[missing] SFT seed={seed} variant={variant['run_tag']}")
                continue
            metrics = load_localnewsqa_metrics(path)
            for split in ("Overall", "Explicit", "Ambiguous"):
                per_split_values[split].append(metrics[f"{split}_accuracy"])
                total_rows[split] = int(metrics[f"{split}_total"])

        for split in ("Explicit", "Ambiguous", "Overall"):
            values = per_split_values[split]
            if not values:
                continue
            rows.append(
                {
                    "family": variant["family"],
                    "series": variant["series"],
                    "split": split,
                    "accuracy": mean(values),
                    "accuracy_std": sample_std(values),
                    "seed_count": len(values),
                    "total": total_rows[split],
                }
            )
    rows.sort(key=lambda row: (row["family"], row["series"], SPLIT_ORDER[row["split"]]))
    return rows


def main() -> int:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    rows = aggregate_sft(args, seeds)
    if rows:
        ensure_parent(args.plot_csv)
        pd.DataFrame(rows).to_csv(args.plot_csv, index=False)
        print(f"[ok] Wrote SFT LocalNewsQA CSV: {args.plot_csv}")
    else:
        print("[warn] No completed SFT LocalNewsQA rows found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
