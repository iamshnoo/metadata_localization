#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


DEFAULT_ROOT = Path(
    "/path/to/metacul/results/downstream_localnewsqa_pretrained_figure9_frozen_full_multiseed"
)
DEFAULT_PLOT_CSV = Path(
    "/path/to/metacul/results/plots/plot8/plot_8_pretrained_target_split_multiseed.csv"
)

VARIANTS = [
    {
        "family": "1B",
        "series": "1B T+/I+",
        "root_name": "1b_codeg_labels_question_final",
        "run_tag": "1b_tplus_eplus",
        "eval_meta_tag": "with_metadata",
    },
    {
        "family": "1B",
        "series": "1B T+/I-",
        "root_name": "1b_codeg_labels_question_final",
        "run_tag": "1b_tplus_eminus",
        "eval_meta_tag": "without_metadata",
    },
    {
        "family": "1B",
        "series": "1B T-/I+",
        "root_name": "1b_codeg_labels_question_final",
        "run_tag": "1b_tminus_eplus",
        "eval_meta_tag": "with_metadata",
    },
    {
        "family": "1B",
        "series": "1B T-/I-",
        "root_name": "1b_codeg_labels_question_final",
        "run_tag": "1b_tminus_eminus",
        "eval_meta_tag": "without_metadata",
    },
    {
        "family": "3B",
        "series": "3B T+/I+",
        "root_name": "3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos",
        "run_tag": "3b_tplus_eplus",
        "eval_meta_tag": "with_metadata",
    },
    {
        "family": "3B",
        "series": "3B T+/I-",
        "root_name": "3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos",
        "run_tag": "3b_tplus_eminus",
        "eval_meta_tag": "without_metadata",
    },
    {
        "family": "3B",
        "series": "3B T-/I+",
        "root_name": "3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos",
        "run_tag": "3b_tminus_eplus",
        "eval_meta_tag": "with_metadata",
    },
    {
        "family": "3B",
        "series": "3B T-/I-",
        "root_name": "3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos",
        "run_tag": "3b_tminus_eminus",
        "eval_meta_tag": "without_metadata",
    },
]

SPLIT_ORDER = {"Overall": 0, "Explicit": 1, "Ambiguous": 2}
FAMILY_SCORING = {
    "1B": {"alpha": 1.25, "beta": 0.0},
    "3B": {"alpha": 0.25, "beta": 0.5},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate frozen raw-pretrained LocalNewsQA multiseed results."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seeds", default="41,42,43,44,45")
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


def option_lengths_from_row(row: Dict) -> Optional[List[float]]:
    sums = row.get("option_loglikelihood_sums")
    avgs = row.get("option_loglikelihood_avgs")
    if not isinstance(sums, list) or not isinstance(avgs, list) or len(sums) != len(avgs):
        return None
    lengths: List[float] = []
    for s, a in zip(sums, avgs):
        s_f = float(s)
        a_f = float(a)
        lengths.append(max(1.0, abs(s_f / a_f)) if a_f != 0 else 1.0)
    return lengths


def rescored_prediction(row: Dict, alpha: float, beta: float) -> Optional[Tuple[int, int]]:
    sums = row.get("option_loglikelihood_sums")
    avgs = row.get("option_loglikelihood_avgs")
    opts = row.get("prompt_options")
    gold = row.get("eval_correct_answer")
    if not isinstance(sums, list) or not isinstance(avgs, list) or not isinstance(opts, list):
        return None
    if gold not in opts:
        return None
    lengths = option_lengths_from_row(row)
    if lengths is None:
        return None
    primary = [float(s) / ((length ** alpha) if alpha != 0 else 1.0) for s, length in zip(sums, lengths)]
    raw_calib = row.get("null_calibration_option_loglikelihood_sums")
    raw_calib_avgs = row.get("null_calibration_option_loglikelihood_avgs")
    if raw_calib and raw_calib_avgs and beta != 0:
        calib_lengths = []
        for cal_sum, cal_avg in zip(raw_calib, raw_calib_avgs):
            cal_sum_f = float(cal_sum)
            cal_avg_f = float(cal_avg)
            calib_lengths.append(max(1.0, abs(cal_sum_f / cal_avg_f)) if cal_avg_f != 0 else 1.0)
        calibration = [
            float(cal_sum) / ((length ** alpha) if alpha != 0 else 1.0)
            for cal_sum, length in zip(raw_calib, calib_lengths)
        ]
        scores = [p - beta * c for p, c in zip(primary, calibration)]
    else:
        scores = primary
    pred_idx = max(range(len(scores)), key=lambda i: scores[i])
    gold_idx = opts.index(gold)
    return pred_idx, gold_idx


def load_localnewsqa_metrics(path: Path, family: str) -> Dict[str, float]:
    counts = {
        "Overall": {"correct": 0, "total": 0},
        "Explicit": {"correct": 0, "total": 0},
        "Ambiguous": {"correct": 0, "total": 0},
    }
    scoring = FAMILY_SCORING[family]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            split_type = str(row.get("split_type", "")).strip().lower()
            split_name = "Ambiguous" if split_type == "ambiguous" else "Explicit"
            pred = rescored_prediction(row, alpha=scoring["alpha"], beta=scoring["beta"])
            if pred is None:
                if "is_correct" not in row:
                    continue
                is_correct = bool(row["is_correct"])
            else:
                pred_idx, gold_idx = pred
                is_correct = pred_idx == gold_idx
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


def aggregate(args: argparse.Namespace, seeds: List[int]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for variant in VARIANTS:
        per_split_values: Dict[str, List[float]] = {
            "Overall": [],
            "Explicit": [],
            "Ambiguous": [],
        }
        total_rows = {"Overall": None, "Explicit": None, "Ambiguous": None}
        for seed in seeds:
            seed_dir = args.root / variant["root_name"] / f"seed_{seed}"
            path = find_single(
                f"localnewsqa_eval_target_{variant['eval_meta_tag']}_custom*{variant['run_tag']}_seed{seed}_c0*.jsonl",
                seed_dir,
            )
            if path is None:
                print(f"[missing] raw seed={seed} variant={variant['run_tag']}")
                continue
            metrics = load_localnewsqa_metrics(path, family=variant["family"])
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
    rows = aggregate(args, seeds)
    if rows:
        ensure_parent(args.plot_csv)
        pd.DataFrame(rows).to_csv(args.plot_csv, index=False)
        print(f"[ok] Wrote frozen raw LocalNewsQA CSV: {args.plot_csv}")
    else:
        print("[warn] No completed frozen raw LocalNewsQA rows found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
