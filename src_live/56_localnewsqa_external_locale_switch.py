#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


MODEL_ORDER = [
    ("llama32_1b", "LLaMA-3.2-1B"),
    ("llama32_3b", "LLaMA-3.2-3B"),
    ("qwen25_0p5b", "Qwen2.5-0.5B"),
    ("qwen25_1p5b", "Qwen2.5-1.5B"),
    ("qwen25_3b", "Qwen2.5-3B"),
    ("qwen35_0p8b", "Qwen3.5-0.8B"),
    ("qwen35_2b", "Qwen3.5-2B"),
    ("qwen35_4b", "Qwen3.5-4B"),
    ("qwen35_9b", "Qwen3.5-9B"),
    ("gemma4_e2b_it", "Gemma-4-E2B-it"),
    ("gemma4_e4b_it", "Gemma-4-E4B-it"),
    ("mistral7b_v02", "Mistral-7B-Instruct-v0.2"),
]
META_ORDER = ["with_metadata", "without_metadata"]
META_LABEL = {
    "with_metadata": "With metadata",
    "without_metadata": "Without metadata",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute paired locale-switch metrics for external instruct baselines on LocalNewsQA."
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path(
            "/path/to/metacul/results/downstream_localnewsqa_external_baselines_multiseed"
        ),
    )
    parser.add_argument(
        "--contrast-root",
        type=Path,
        default=Path(
            "/path/to/metacul/results/downstream_localnewsqa_external_baselines_contrast_multiseed"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/path/to/metacul/results/analysis/localnewsqa_external_locale_switch"
        ),
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


def answer_index(answer: str, options: Sequence[str]) -> Optional[int]:
    want = normalize_text(answer)
    for idx, opt in enumerate(options):
        if normalize_text(opt) == want:
            return idx
    return None


def row_scores(row: Dict) -> Optional[List[float]]:
    selected = row.get("option_loglikelihood_selected_scores")
    if isinstance(selected, list) and selected:
        return [float(x) for x in selected]
    avgs = row.get("option_loglikelihood_avgs")
    if isinstance(avgs, list) and avgs:
        return [float(x) for x in avgs]
    sums = row.get("option_loglikelihood_sums")
    if isinstance(sums, list) and sums:
        return [float(x) for x in sums]
    return None


def row_prediction(row: Dict) -> Optional[Dict[str, object]]:
    options = list(row.get("prompt_options") or row.get("options") or [])
    if not options:
        return None
    gold_answer = row.get("eval_correct_answer")
    if gold_answer is None:
        return None
    gold_idx = answer_index(str(gold_answer), options)
    if gold_idx is None:
        return None
    scores = row_scores(row)
    if scores is None or len(scores) != len(options):
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


def collect_model(
    target_root: Path,
    contrast_root: Path,
    model_slug: str,
    model_label: str,
    expected_seeds: Sequence[int],
) -> pd.DataFrame:
    records: List[Dict] = []
    pair_counts: Dict[Tuple[int, str], int] = {}

    for seed in expected_seeds:
        for meta_tag in META_ORDER:
            target_path = (
                target_root
                / f"seed_{seed}"
                / f"localnewsqa_{model_slug}_{meta_tag}_target.jsonl"
            )
            contrast_path = (
                contrast_root
                / f"seed_{seed}"
                / f"localnewsqa_{model_slug}_{meta_tag}_contrast.jsonl"
            )
            if not target_path.exists() or not contrast_path.exists():
                raise FileNotFoundError(
                    f"Missing expected JSONLs for {model_label} seed={seed} meta={meta_tag}: "
                    f"target={target_path.exists()} contrast={contrast_path.exists()}"
                )

            pair_count = 0
            target_rows = {
                safe_item_id(row): row
                for row in load_jsonl(target_path)
                if normalize_text(row.get("split_type")) == "ambiguous"
            }
            contrast_rows = {
                safe_item_id(row): row
                for row in load_jsonl(contrast_path)
                if normalize_text(row.get("split_type")) == "ambiguous"
            }

            for item_id in sorted(set(target_rows) & set(contrast_rows)):
                target_row = target_rows[item_id]
                contrast_row = contrast_rows[item_id]
                target_pred = row_prediction(target_row)
                contrast_pred = row_prediction(contrast_row)
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
                        "model_slug": model_slug,
                        "model": model_label,
                        "seed": seed,
                        "meta_tag": meta_tag,
                        "meta_label": META_LABEL[meta_tag],
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

            pair_counts[(seed, meta_tag)] = pair_count

    if not records:
        raise FileNotFoundError(f"No usable paired ambiguous rows found for {model_label}")
    unique_counts = sorted(set(pair_counts.values()))
    if len(unique_counts) != 1:
        raise ValueError(f"Inconsistent paired switch counts for {model_label}: {pair_counts}")
    return pd.DataFrame(records)


def summarize_model(df: pd.DataFrame, n_boot: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict] = []
    delta_rows: List[Dict] = []

    per_item = (
        df.groupby(["model_slug", "model", "meta_tag", "meta_label", "item_id"], dropna=False)[
            ["target_correct", "contrast_correct", "exact_pair_success", "margin_switch_success"]
        ]
        .mean()
        .reset_index()
    )

    for meta_tag in META_ORDER:
        sub = per_item.loc[per_item["meta_tag"] == meta_tag].copy()
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
                "model_slug": sub["model_slug"].iloc[0],
                "model": sub["model"].iloc[0],
                "meta_tag": meta_tag,
                "meta_label": META_LABEL[meta_tag],
                "n_switch_pairs": int(len(sub)),
                "seed_count": int(df.loc[df["meta_tag"] == meta_tag, "seed"].nunique()),
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

    plus = per_item.loc[per_item["meta_tag"] == "with_metadata"].copy()
    minus = per_item.loc[per_item["meta_tag"] == "without_metadata"].copy()
    joined = plus.merge(
        minus,
        on=["model_slug", "model", "item_id"],
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
                "model_slug": joined["model_slug"].iloc[0],
                "model": joined["model"].iloc[0],
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

    model_frames = [
        collect_model(
            args.target_root,
            args.contrast_root,
            model_slug,
            model_label,
            expected_seeds=args.expected_seeds,
        )
        for model_slug, model_label in MODEL_ORDER
    ]
    pair_df = pd.concat(model_frames, ignore_index=True)
    pair_df.to_csv(args.output_root / "pair_level.csv", index=False)

    summaries: List[pd.DataFrame] = []
    deltas: List[pd.DataFrame] = []
    for model_slug, model_label in MODEL_ORDER:
        model_df = pair_df.loc[pair_df["model_slug"] == model_slug].copy()
        summary_df, delta_df = summarize_model(
            model_df, n_boot=args.bootstrap_samples, seed=args.bootstrap_seed
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
