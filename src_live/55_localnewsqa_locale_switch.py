#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


VARIANT_ORDER = ["tplus_eplus", "tminus_eminus"]
VARIANT_LABEL = {
    "tplus_eplus": "(T+, I+)",
    "tminus_eminus": "(T-, I-)",
}
FAMILY_SCORING = {
    "MAPLE 1B": {"alpha": 1.25, "beta": 0.0},
    "MAPLE 3B": {"alpha": 0.25, "beta": 0.5},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute direct locale-switch metrics from target and contrast LocalNewsQA runs."
    )
    parser.add_argument(
        "--target-root-1b",
        type=Path,
        default=Path(
            "/path/to/metacul/results/downstream_localnewsqa_pretrained_figure9_frozen_full_multiseed/1b_codeg_labels_question_final"
        ),
    )
    parser.add_argument(
        "--contrast-root-1b",
        type=Path,
        default=Path(
            "/path/to/metacul/results/downstream_localnewsqa_pretrained_figure9_contrast_full_multiseed/1b_codeg_labels_question_final"
        ),
    )
    parser.add_argument(
        "--target-root-3b",
        type=Path,
        default=Path(
            "/path/to/metacul/results/downstream_localnewsqa_pretrained_figure9_frozen_full_multiseed/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos"
        ),
    )
    parser.add_argument(
        "--contrast-root-3b",
        type=Path,
        default=Path(
            "/path/to/metacul/results/downstream_localnewsqa_pretrained_figure9_contrast_full_multiseed/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/path/to/metacul/results/analysis/localnewsqa_locale_switch"),
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        nargs="+",
        default=[41, 42, 43, 44, 45],
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def detect_variant(path: Path) -> Optional[str]:
    name = path.name
    for variant in VARIANT_ORDER:
        if variant in name:
            return variant
    return None


def safe_item_id(row: Dict) -> Tuple:
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
    for s, a in zip(sums, avgs):
        s_f = float(s)
        a_f = float(a)
        lengths.append(max(1.0, abs(s_f / a_f)) if a_f != 0 else 1.0)
    return lengths


def rescored_scores(row: Dict, family: str) -> Optional[List[float]]:
    sums = row.get("option_loglikelihood_sums")
    if not isinstance(sums, list):
        return None
    lengths = option_lengths(row)
    if lengths is None:
        return None
    scoring = FAMILY_SCORING[family]
    alpha = scoring["alpha"]
    beta = scoring["beta"]
    primary = [float(s) / ((length ** alpha) if alpha != 0 else 1.0) for s, length in zip(sums, lengths)]
    raw_calib = row.get("null_calibration_option_loglikelihood_sums")
    raw_calib_avgs = row.get("null_calibration_option_loglikelihood_avgs")
    if raw_calib and raw_calib_avgs and beta != 0:
        calib_lengths: List[float] = []
        for cal_sum, cal_avg in zip(raw_calib, raw_calib_avgs):
            cal_sum_f = float(cal_sum)
            cal_avg_f = float(cal_avg)
            calib_lengths.append(max(1.0, abs(cal_sum_f / cal_avg_f)) if cal_avg_f != 0 else 1.0)
        calibration = [
            float(cal_sum) / ((length ** alpha) if alpha != 0 else 1.0)
            for cal_sum, length in zip(raw_calib, calib_lengths)
        ]
        return [p - beta * c for p, c in zip(primary, calibration)]
    return primary


def answer_index(answer: str, options: Sequence[str]) -> Optional[int]:
    want = normalize_text(answer)
    for idx, opt in enumerate(options):
        if normalize_text(opt) == want:
            return idx
    return None


def row_prediction(row: Dict, family: str) -> Optional[Dict[str, object]]:
    options = list(row.get("prompt_options") or row.get("options") or [])
    if not options:
        return None
    gold_answer = row.get("eval_correct_answer")
    if gold_answer is None:
        return None
    gold_idx = answer_index(str(gold_answer), options)
    if gold_idx is None:
        return None
    scores = rescored_scores(row, family=family)
    if scores is None:
        return None
    pred_idx = int(np.argmax(np.asarray(scores, dtype=float)))
    return {
        "gold_idx": gold_idx,
        "pred_idx": pred_idx,
        "scores": scores,
        "is_correct": float(pred_idx == gold_idx),
        "gold_answer": str(gold_answer),
        "pred_answer": str(options[pred_idx]),
    }


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def collect_family(
    target_root: Path,
    contrast_root: Path,
    family: str,
    expected_seeds: Sequence[int],
) -> pd.DataFrame:
    records: List[Dict] = []
    target_map: Dict[Tuple[int, str], Path] = {}
    contrast_map: Dict[Tuple[int, str], Path] = {}

    for path in sorted(target_root.rglob("*.jsonl")):
        variant = detect_variant(path)
        if variant is None:
            continue
        match = re.search(r"seed_(\d+)", str(path))
        seed = int(match.group(1)) if match else -1
        target_map[(seed, variant)] = path
    for path in sorted(contrast_root.rglob("*.jsonl")):
        variant = detect_variant(path)
        if variant is None:
            continue
        match = re.search(r"seed_(\d+)", str(path))
        seed = int(match.group(1)) if match else -1
        contrast_map[(seed, variant)] = path

    expected_keys = {(seed, variant) for seed in expected_seeds for variant in VARIANT_ORDER}
    missing_target = sorted(expected_keys - set(target_map))
    missing_contrast = sorted(expected_keys - set(contrast_map))
    if missing_target or missing_contrast:
        raise FileNotFoundError(
            f"Missing expected JSONLs for {family}. "
            f"target_missing={missing_target} contrast_missing={missing_contrast}"
        )

    overlapping_keys = sorted(expected_keys)
    pair_counts: Dict[Tuple[int, str], int] = {}

    for seed, variant in overlapping_keys:
        pair_count = 0
        target_rows = {
            safe_item_id(row): row
            for row in load_jsonl(target_map[(seed, variant)])
            if normalize_text(row.get("split_type")) == "ambiguous"
        }
        contrast_rows = {
            safe_item_id(row): row
            for row in load_jsonl(contrast_map[(seed, variant)])
            if normalize_text(row.get("split_type")) == "ambiguous"
        }
        for item_id in sorted(set(target_rows) & set(contrast_rows)):
            target_row = target_rows[item_id]
            contrast_row = contrast_rows[item_id]
            target_pred = row_prediction(target_row, family=family)
            contrast_pred = row_prediction(contrast_row, family=family)
            if target_pred is None or contrast_pred is None:
                continue

            same_gold = normalize_text(target_pred["gold_answer"]) == normalize_text(
                contrast_pred["gold_answer"]
            )
            if same_gold:
                continue

            target_scores = target_pred["scores"]
            contrast_scores = contrast_pred["scores"]
            target_gold_idx = int(target_pred["gold_idx"])
            contrast_gold_idx = int(contrast_pred["gold_idx"])
            target_margin = float(target_scores[target_gold_idx] - target_scores[contrast_gold_idx])
            contrast_margin = float(
                contrast_scores[contrast_gold_idx] - contrast_scores[target_gold_idx]
            )

            records.append(
                {
                    "family": family,
                    "seed": seed,
                    "variant": variant,
                    "variant_label": VARIANT_LABEL[variant],
                    "item_id": str(item_id),
                    "country": str(target_row.get("country") or ""),
                    "contrast_country": str(target_row.get("contrast_country") or ""),
                    "topic": str(target_row.get("topic") or ""),
                    "year": str(target_row.get("year") or ""),
                    "target_correct": float(target_pred["is_correct"]),
                    "contrast_correct": float(contrast_pred["is_correct"]),
                    "exact_pair_success": float(
                        bool(target_pred["is_correct"]) and bool(contrast_pred["is_correct"])
                    ),
                    "margin_switch_success": float(target_margin > 0.0 and contrast_margin > 0.0),
                    "target_margin": target_margin,
                    "contrast_margin": contrast_margin,
                }
            )
            pair_count += 1
        pair_counts[(seed, variant)] = pair_count

    if not records:
        raise FileNotFoundError(f"No usable paired ambiguous rows found for {family}")
    unique_counts = sorted(set(pair_counts.values()))
    if len(unique_counts) != 1:
        raise ValueError(
            f"Inconsistent paired switch counts for {family}: {pair_counts}"
        )
    return pd.DataFrame(records)


def summarize_family(df: pd.DataFrame, n_boot: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict] = []
    delta_rows: List[Dict] = []

    per_item = (
        df.groupby(["family", "variant", "variant_label", "item_id"], dropna=False)[
            ["target_correct", "contrast_correct", "exact_pair_success", "margin_switch_success"]
        ]
        .mean()
        .reset_index()
    )

    for variant in VARIANT_ORDER:
        sub = per_item.loc[per_item["variant"] == variant].copy()
        if sub.empty:
            continue
        target_vals = sub["target_correct"].to_numpy(dtype=float)
        contrast_vals = sub["contrast_correct"].to_numpy(dtype=float)
        exact_vals = sub["exact_pair_success"].to_numpy(dtype=float)
        margin_vals = sub["margin_switch_success"].to_numpy(dtype=float)
        target_ci = bootstrap_ci(target_vals, n_boot=n_boot, seed=seed + 1)
        contrast_ci = bootstrap_ci(contrast_vals, n_boot=n_boot, seed=seed + 2)
        exact_ci = bootstrap_ci(exact_vals, n_boot=n_boot, seed=seed + 3)
        margin_ci = bootstrap_ci(margin_vals, n_boot=n_boot, seed=seed + 4)
        summary_rows.append(
            {
                "family": sub["family"].iloc[0],
                "variant": variant,
                "variant_label": VARIANT_LABEL[variant],
                "n_switch_pairs": int(len(sub)),
                "seed_count": int(df.loc[df["variant"] == variant, "seed"].nunique()),
                "target_accuracy_pct": float(target_vals.mean() * 100.0),
                "target_ci_low_pct": float(target_ci[0] * 100.0),
                "target_ci_high_pct": float(target_ci[1] * 100.0),
                "contrast_accuracy_pct": float(contrast_vals.mean() * 100.0),
                "contrast_ci_low_pct": float(contrast_ci[0] * 100.0),
                "contrast_ci_high_pct": float(contrast_ci[1] * 100.0),
                "exact_pair_accuracy_pct": float(exact_vals.mean() * 100.0),
                "exact_pair_ci_low_pct": float(exact_ci[0] * 100.0),
                "exact_pair_ci_high_pct": float(exact_ci[1] * 100.0),
                "margin_switch_rate_pct": float(margin_vals.mean() * 100.0),
                "margin_switch_ci_low_pct": float(margin_ci[0] * 100.0),
                "margin_switch_ci_high_pct": float(margin_ci[1] * 100.0),
            }
        )

    plus = per_item.loc[per_item["variant"] == "tplus_eplus"].copy()
    minus = per_item.loc[per_item["variant"] == "tminus_eminus"].copy()
    joined = plus.merge(
        minus,
        on=["family", "item_id"],
        suffixes=("_plus", "_minus"),
    )
    if not joined.empty:
        exact_delta = (
            joined["exact_pair_success_plus"].to_numpy(dtype=float)
            - joined["exact_pair_success_minus"].to_numpy(dtype=float)
        )
        margin_delta = (
            joined["margin_switch_success_plus"].to_numpy(dtype=float)
            - joined["margin_switch_success_minus"].to_numpy(dtype=float)
        )
        exact_ci = bootstrap_ci(exact_delta, n_boot=n_boot, seed=seed + 10)
        margin_ci = bootstrap_ci(margin_delta, n_boot=n_boot, seed=seed + 11)
        delta_rows.append(
            {
                "family": joined["family"].iloc[0],
                "n_switch_pairs": int(len(joined)),
                "exact_pair_delta_pp": float(exact_delta.mean() * 100.0),
                "exact_pair_ci_low_pp": float(exact_ci[0] * 100.0),
                "exact_pair_ci_high_pp": float(exact_ci[1] * 100.0),
                "margin_switch_delta_pp": float(margin_delta.mean() * 100.0),
                "margin_switch_ci_low_pp": float(margin_ci[0] * 100.0),
                "margin_switch_ci_high_pp": float(margin_ci[1] * 100.0),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(delta_rows)


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    family_frames = [
        collect_family(
            args.target_root_1b,
            args.contrast_root_1b,
            "MAPLE 1B",
            expected_seeds=args.expected_seeds,
        ),
        collect_family(
            args.target_root_3b,
            args.contrast_root_3b,
            "MAPLE 3B",
            expected_seeds=args.expected_seeds,
        ),
    ]
    pair_df = pd.concat(family_frames, ignore_index=True)
    pair_df.to_csv(args.output_root / "pair_level.csv", index=False)

    summaries: List[pd.DataFrame] = []
    deltas: List[pd.DataFrame] = []
    for fam in ["MAPLE 1B", "MAPLE 3B"]:
        fam_df = pair_df.loc[pair_df["family"] == fam].copy()
        summary_df, delta_df = summarize_family(
            fam_df, n_boot=args.bootstrap_samples, seed=args.bootstrap_seed
        )
        summaries.append(summary_df)
        deltas.append(delta_df)

    summary_df = pd.concat(summaries, ignore_index=True)
    delta_df = pd.concat(deltas, ignore_index=True)
    summary_df.to_csv(args.output_root / "summary.csv", index=False)
    delta_df.to_csv(args.output_root / "deltas.csv", index=False)

    payload = {
        "pair_level_csv": str(args.output_root / "pair_level.csv"),
        "summary_csv": str(args.output_root / "summary.csv"),
        "deltas_csv": str(args.output_root / "deltas.csv"),
    }
    print(summary_df.to_string(index=False))
    print(delta_df.to_string(index=False))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
