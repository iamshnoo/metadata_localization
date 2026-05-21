#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path("/path/to/metacul")
DEFAULT_INPUT_ROOT = ROOT / "results/downstream_localnewsqa_pretrained_figure9_frozen_full_multiseed"
DEFAULT_OUTPUT_CSV = ROOT / "results/plots/plot8/plot_8_pretrained_target_split_seed41_bootstrap.csv"

VARIANTS = [
    ("1B", "1B T+/I+", "1b_codeg_labels_question_final", "1b_tplus_eplus", "with_metadata"),
    ("1B", "1B T+/I-", "1b_codeg_labels_question_final", "1b_tplus_eminus", "without_metadata"),
    ("1B", "1B T-/I+", "1b_codeg_labels_question_final", "1b_tminus_eplus", "with_metadata"),
    ("1B", "1B T-/I-", "1b_codeg_labels_question_final", "1b_tminus_eminus", "without_metadata"),
    ("3B", "3B T+/I+", "3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos", "3b_tplus_eplus", "with_metadata"),
    ("3B", "3B T+/I-", "3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos", "3b_tplus_eminus", "without_metadata"),
    ("3B", "3B T-/I+", "3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos", "3b_tminus_eplus", "with_metadata"),
    ("3B", "3B T-/I-", "3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos", "3b_tminus_eminus", "without_metadata"),
]

SPLIT_ORDER = {"Overall": 0, "Explicit": 1, "Ambiguous": 2}
FAMILY_SCORING = {
    "1B": {"alpha": 1.25, "beta": 0.0},
    "3B": {"alpha": 0.25, "beta": 0.5},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build seed-41 question-bootstrap LocalNewsQA rows for Figure 2."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def item_key(row: Dict) -> Tuple[str, str, str, str, str, str, str]:
    return (
        str(row.get("generation_custom_id") or ""),
        normalize_text(row.get("question")),
        normalize_text(row.get("country")),
        normalize_text(row.get("target_country")),
        normalize_text(row.get("contrast_country")),
        normalize_text(row.get("topic")),
        str(row.get("year") or ""),
    )


def option_lengths(row: Dict) -> Optional[List[float]]:
    sums = row.get("option_loglikelihood_sums")
    avgs = row.get("option_loglikelihood_avgs")
    if not isinstance(sums, list) or not isinstance(avgs, list) or len(sums) != len(avgs):
        return None
    lengths: List[float] = []
    for score_sum, avg in zip(sums, avgs):
        score_sum = float(score_sum)
        avg = float(avg)
        lengths.append(max(1.0, abs(score_sum / avg)) if avg != 0 else 1.0)
    return lengths


def answer_index(answer: str, options: Sequence[str]) -> Optional[int]:
    want = normalize_text(answer)
    for idx, option in enumerate(options):
        if normalize_text(option) == want:
            return idx
    return None


def is_correct(row: Dict, family: str) -> Optional[float]:
    options = list(row.get("prompt_options") or row.get("options") or [])
    gold_answer = row.get("eval_correct_answer")
    sums = row.get("option_loglikelihood_sums")
    if not options or gold_answer is None or not isinstance(sums, list):
        return None
    gold_idx = answer_index(str(gold_answer), options)
    lengths = option_lengths(row)
    if gold_idx is None or lengths is None or len(sums) != len(options):
        return None
    scoring = FAMILY_SCORING[family]
    alpha = scoring["alpha"]
    beta = scoring["beta"]
    primary = [float(score_sum) / (length ** alpha) for score_sum, length in zip(sums, lengths)]
    raw_calib = row.get("null_calibration_option_loglikelihood_sums")
    raw_calib_avgs = row.get("null_calibration_option_loglikelihood_avgs")
    if raw_calib and raw_calib_avgs and beta != 0:
        calib_lengths = []
        for cal_sum, cal_avg in zip(raw_calib, raw_calib_avgs):
            cal_sum_f = float(cal_sum)
            cal_avg_f = float(cal_avg)
            calib_lengths.append(max(1.0, abs(cal_sum_f / cal_avg_f)) if cal_avg_f != 0 else 1.0)
        calibration = [
            float(cal_sum) / (length ** alpha)
            for cal_sum, length in zip(raw_calib, calib_lengths)
        ]
        scores = [p - beta * c for p, c in zip(primary, calibration)]
    else:
        scores = primary
    return float(int(np.argmax(np.asarray(scores, dtype=float))) == gold_idx)


def bootstrap_ci(values: Sequence[float], n_boot: int, seed: int) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def find_input(root: Path, seed: int, root_name: str, run_tag: str, eval_meta_tag: str) -> Path:
    seed_dir = root / root_name / f"seed_{seed}"
    matches = sorted(
        seed_dir.glob(f"localnewsqa_eval_target_{eval_meta_tag}_custom*{run_tag}_seed{seed}_c0*.jsonl")
    )
    if not matches:
        raise FileNotFoundError(f"Missing LocalNewsQA input for {run_tag} seed {seed}")
    return matches[0]


def load_split_values(path: Path, family: str) -> Dict[str, Dict[Tuple[str, str, str, str, str, str, str], float]]:
    values: Dict[str, Dict[Tuple[str, str, str, str, str, str, str], float]] = {
        "Overall": {},
        "Explicit": {},
        "Ambiguous": {},
    }
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            correct = is_correct(row, family)
            if correct is None:
                continue
            split = "Ambiguous" if normalize_text(row.get("split_type")) == "ambiguous" else "Explicit"
            key = item_key(row)
            values["Overall"][key] = correct
            values[split][key] = correct
    return values


def main() -> int:
    args = parse_args()
    rows = []
    for family, series, root_name, run_tag, eval_meta_tag in VARIANTS:
        path = find_input(args.input_root, args.seed, root_name, run_tag, eval_meta_tag)
        split_values = load_split_values(path, family)
        for split in ("Overall", "Explicit", "Ambiguous"):
            vals = list(split_values[split].values())
            lo, hi = bootstrap_ci(
                vals,
                n_boot=args.bootstrap_samples,
                seed=args.bootstrap_seed + len(rows),
            )
            rows.append(
                {
                    "family": family,
                    "series": series,
                    "split": split,
                    "accuracy": float(np.mean(vals)),
                    "ci_low": lo,
                    "ci_high": hi,
                    "seed": args.seed,
                    "total": len(vals),
                    "source": str(path),
                }
            )
    rows.sort(key=lambda row: (row["family"], row["series"], SPLIT_ORDER[row["split"]]))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"[ok] wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
