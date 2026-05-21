#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT_1B = Path(
    "/path/to/metacul/results/downstream_localnewsqa_pretrained_figure9_frozen_full_multiseed/1b_codeg_labels_question_final"
)
DEFAULT_ROOT_3B = Path(
    "/path/to/metacul/results/downstream_localnewsqa_pretrained_figure9_frozen_full_multiseed/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos"
)
DEFAULT_VALIDATION_CSV = Path(
    "/path/to/metacul/qa_data/localnewsqa_core/runs/human_validation_ambiguous_510_web.csv"
)
DEFAULT_OUTPUT_ROOT = Path("/path/to/metacul/results/analysis/localnewsqa_significance")

VARIANT_ORDER = ["tplus_eplus", "tplus_eminus", "tminus_eplus", "tminus_eminus"]
VARIANT_LABEL = {
    "tplus_eplus": "(T+, I+)",
    "tplus_eminus": "(T+, I-)",
    "tminus_eplus": "(T-, I+)",
    "tminus_eminus": "(T-, I-)",
}
SPLIT_ORDER = ["overall", "explicit", "ambiguous", "validated_ambiguous"]
FAMILY_SCORING = {
    "1B": {"alpha": 1.25, "beta": 0.0},
    "3B": {"alpha": 0.25, "beta": 0.5},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Question-level significance, effect sizes, and consistency analysis for LocalNewsQA."
    )
    parser.add_argument("--root-1b", type=Path, default=DEFAULT_ROOT_1B)
    parser.add_argument("--root-3b", type=Path, default=DEFAULT_ROOT_3B)
    parser.add_argument("--validation-csv", type=Path, default=DEFAULT_VALIDATION_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def item_key_from_row(row: Dict) -> Tuple:
    return (
        normalize_text(row.get("country")),
        normalize_text(row.get("contrast_country")),
        normalize_text(row.get("question")),
        normalize_text(row.get("topic")),
        str(row.get("year") or ""),
        normalize_text(row.get("correct_answer") or row.get("eval_correct_answer")),
        normalize_text(row.get("split_type")),
    )


def item_key_from_validation(row: pd.Series) -> Tuple:
    return (
        normalize_text(row.get("country")),
        normalize_text(row.get("contrast_country")),
        normalize_text(row.get("question")),
        normalize_text(row.get("topic")),
        str(row.get("year") or ""),
        normalize_text(row.get("target_answer")),
        "ambiguous",
    )


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


def score_list_from_row(row: Dict, family: str) -> Optional[List[float]]:
    sums = row.get("option_loglikelihood_sums")
    if isinstance(sums, list):
        lengths = option_lengths_from_row(row)
        if lengths is not None:
            scoring = FAMILY_SCORING[family]
            raw_calib = row.get("null_calibration_option_loglikelihood_sums")
            calib = [0.0] * len(sums) if raw_calib is None else [float(x) for x in raw_calib]
            return [
                float(s) / ((length ** scoring["alpha"]) if scoring["alpha"] != 0 else 1.0)
                - scoring["beta"] * cal
                for s, length, cal in zip(sums, lengths, calib)
            ]
    for key in (
        "option_loglikelihood_selected_scores",
        "option_scores",
        "option_loglikelihood_primary_scores",
        "option_loglikelihood_avgs",
    ):
        value = row.get(key)
        if isinstance(value, list) and value:
            return [float(x) for x in value]
    return None


def find_correct_index(row: Dict) -> Optional[int]:
    correct = normalize_text(row.get("eval_correct_answer") or row.get("correct_answer"))
    if not correct:
        return None
    for key in ("prompt_options", "scoring_candidates", "options"):
        opts = row.get(key)
        if isinstance(opts, list) and opts:
            normalized = [normalize_text(str(x).lstrip()) for x in opts]
            for idx, opt in enumerate(normalized):
                if opt == correct:
                    return idx
    return None


def compute_margins(row: Dict, family: str) -> Tuple[Optional[float], Optional[float]]:
    scores = score_list_from_row(row, family=family)
    if not scores or len(scores) < 2:
        return None, None
    ordered = sorted(scores, reverse=True)
    top_margin = float(ordered[0] - ordered[1])
    correct_idx = find_correct_index(row)
    if correct_idx is None:
        return top_margin, None
    correct_score = float(scores[correct_idx])
    other_scores = [float(scores[i]) for i in range(len(scores)) if i != correct_idx]
    if not other_scores:
        return top_margin, None
    correct_margin = float(correct_score - max(other_scores))
    return top_margin, correct_margin


def load_family_rows(root: Path, family: str) -> pd.DataFrame:
    records: List[Dict] = []
    for jsonl_path in sorted(root.rglob("*.jsonl")):
        variant = detect_variant(jsonl_path)
        if variant is None:
            continue
        seed_match = re.search(r"seed_(\d+)", str(jsonl_path))
        seed = int(seed_match.group(1)) if seed_match else -1
        for row in load_jsonl(jsonl_path):
            key = item_key_from_row(row)
            top_margin, correct_margin = compute_margins(row, family=family)
            scored = score_list_from_row(row, family=family)
            correct_idx = find_correct_index(row)
            if scored is None or correct_idx is None:
                is_correct = float(bool(row.get("is_correct")))
            else:
                pred_idx = int(np.argmax(np.asarray(scored, dtype=float)))
                is_correct = float(pred_idx == correct_idx)
            records.append(
                {
                    "family": family,
                    "variant": variant,
                    "seed": seed,
                    "item_key": key,
                    "country": str(row.get("country") or ""),
                    "contrast_country": str(row.get("contrast_country") or ""),
                    "topic": str(row.get("topic") or ""),
                    "split_type": str(row.get("split_type") or "").lower(),
                    "year": str(row.get("year") or ""),
                    "is_correct": is_correct,
                    "top_margin": top_margin,
                    "correct_margin": correct_margin,
                }
            )
    if not records:
        raise FileNotFoundError(f"No JSONL rows found under {root}")
    df = pd.DataFrame(records)
    numeric_cols = ["is_correct", "top_margin", "correct_margin"]
    item_df = (
        df.groupby(
            [
                "family",
                "variant",
                "item_key",
                "country",
                "contrast_country",
                "topic",
                "split_type",
                "year",
            ],
            dropna=False,
        )[numeric_cols]
        .mean()
        .reset_index()
    )
    return item_df


def load_validated_keys(validation_csv: Path) -> set:
    if not validation_csv.exists():
        return set()
    df = pd.read_csv(validation_csv)
    needed = ["judge_target_factuality", "judge_locale_dependence", "judge_no_explicit_leakage"]
    for col in needed:
        if col not in df.columns:
            return set()
    mask = np.ones(len(df), dtype=bool)
    for col in needed:
        mask &= df[col].astype(str).str.strip().str.lower().eq("yes").to_numpy()
    good = df.loc[mask].copy()
    if good.empty:
        return set()
    return {item_key_from_validation(row) for _, row in good.iterrows()}


def subset_mask(df: pd.DataFrame, split_name: str, validated_keys: set) -> pd.Series:
    if split_name == "overall":
        return pd.Series(np.ones(len(df), dtype=bool), index=df.index)
    if split_name == "explicit":
        return df["split_type"].eq("explicit")
    if split_name == "ambiguous":
        return df["split_type"].eq("ambiguous")
    if split_name == "validated_ambiguous":
        return df["split_type"].eq("ambiguous") & df["item_key"].isin(validated_keys)
    raise ValueError(f"Unsupported split: {split_name}")


def bootstrap_means(values: np.ndarray, n_boot: int, seed: int, chunk: int = 200) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return np.array([], dtype=float)
    out = np.empty(n_boot, dtype=float)
    written = 0
    while written < n_boot:
        take = min(chunk, n_boot - written)
        idx = rng.integers(0, n, size=(take, n))
        out[written : written + take] = values[idx].mean(axis=1)
        written += take
    return out


def ci_from_bootstrap(samples: np.ndarray) -> Tuple[float, float]:
    if samples.size == 0:
        return math.nan, math.nan
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def p_value_from_bootstrap(diff_samples: np.ndarray) -> float:
    if diff_samples.size == 0:
        return math.nan
    left = float(np.mean(diff_samples <= 0.0))
    right = float(np.mean(diff_samples >= 0.0))
    return float(min(1.0, 2.0 * min(left, right)))


def accuracy_summary(
    family_df: pd.DataFrame,
    validated_keys: set,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for split_idx, split_name in enumerate(SPLIT_ORDER):
        mask = subset_mask(family_df, split_name, validated_keys)
        split_df = family_df.loc[mask].copy()
        if split_df.empty:
            continue
        for variant in VARIANT_ORDER:
            variant_df = split_df.loc[split_df["variant"] == variant].copy()
            if variant_df.empty:
                continue
            values = variant_df["is_correct"].to_numpy(dtype=float)
            boot = bootstrap_means(
                values=values,
                n_boot=bootstrap_samples,
                seed=bootstrap_seed + split_idx + (37 * (VARIANT_ORDER.index(variant) + 1)),
            )
            ci_low, ci_high = ci_from_bootstrap(boot)
            rows.append(
                {
                    "family": variant_df["family"].iloc[0],
                    "split": split_name,
                    "variant": variant,
                    "accuracy_pct": float(values.mean() * 100.0),
                    "ci_low_pct": float(ci_low * 100.0),
                    "ci_high_pct": float(ci_high * 100.0),
                    "n_items": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def paired_comparisons(
    family_df: pd.DataFrame,
    validated_keys: set,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    comparison_pairs = [
        ("tplus_eplus", "tminus_eminus"),
        ("tplus_eplus", "tplus_eminus"),
        ("tplus_eplus", "tminus_eplus"),
    ]
    rows: List[Dict] = []
    for split_idx, split_name in enumerate(SPLIT_ORDER):
        split_df = family_df.loc[subset_mask(family_df, split_name, validated_keys)].copy()
        if split_df.empty:
            continue
        pivot = split_df.pivot_table(
            index="item_key",
            columns="variant",
            values="is_correct",
            aggfunc="mean",
        )
        if pivot.empty:
            continue
        for pair_idx, (a_name, b_name) in enumerate(comparison_pairs):
            if a_name not in pivot.columns or b_name not in pivot.columns:
                continue
            pair = pivot[[a_name, b_name]].dropna()
            if pair.empty:
                continue
            a = pair[a_name].to_numpy(dtype=float)
            b = pair[b_name].to_numpy(dtype=float)
            diff = a - b
            boot = bootstrap_means(
                values=diff,
                n_boot=bootstrap_samples,
                seed=bootstrap_seed + 1000 + split_idx * 11 + pair_idx,
            )
            ci_low, ci_high = ci_from_bootstrap(boot)
            mean_diff = float(diff.mean())
            baseline_error = max(1e-12, 1.0 - float(b.mean()))
            relative_error_reduction = float((mean_diff) / baseline_error)
            effect_std = float(np.std(diff, ddof=1)) if len(diff) > 1 else math.nan
            rows.append(
                {
                    "family": split_df["family"].iloc[0],
                    "split": split_name,
                    "lhs_variant": a_name,
                    "rhs_variant": b_name,
                    "lhs_accuracy_pct": float(a.mean() * 100.0),
                    "rhs_accuracy_pct": float(b.mean() * 100.0),
                    "delta_pp": float(mean_diff * 100.0),
                    "ci_low_pp": float(ci_low * 100.0),
                    "ci_high_pp": float(ci_high * 100.0),
                    "relative_error_reduction_pct": float(relative_error_reduction * 100.0),
                    "paired_effect_d": float(mean_diff / effect_std) if effect_std and not math.isnan(effect_std) else math.nan,
                    "bootstrap_p": p_value_from_bootstrap(boot),
                    "significant_95": bool(ci_low > 0.0 or ci_high < 0.0),
                    "n_items": int(len(diff)),
                }
            )
    return pd.DataFrame(rows)


def consistency_tables(family_df: pd.DataFrame, validated_keys: set) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows_country: List[Dict] = []
    rows_topic: List[Dict] = []
    rows_summary: List[Dict] = []
    for split_name in ["overall", "ambiguous", "validated_ambiguous"]:
        split_df = family_df.loc[subset_mask(family_df, split_name, validated_keys)].copy()
        if split_df.empty:
            continue
        pivot = split_df.pivot_table(
            index=["item_key", "country", "topic"],
            columns="variant",
            values="is_correct",
            aggfunc="mean",
        ).dropna(subset=["tplus_eplus", "tminus_eminus"])
        if pivot.empty:
            continue
        pivot = pivot.reset_index()
        pivot["delta"] = pivot["tplus_eplus"] - pivot["tminus_eminus"]

        country = (
            pivot.groupby("country", dropna=False)["delta"]
            .mean()
            .reset_index()
            .rename(columns={"country": "group_name", "delta": "delta_pp"})
        )
        country["delta_pp"] = country["delta_pp"] * 100.0
        country["family"] = split_df["family"].iloc[0]
        country["split"] = split_name
        country["group_type"] = "country"
        rows_country.extend(country.to_dict(orient="records"))

        topic = (
            pivot.groupby("topic", dropna=False)["delta"]
            .mean()
            .reset_index()
            .rename(columns={"topic": "group_name", "delta": "delta_pp"})
        )
        topic["delta_pp"] = topic["delta_pp"] * 100.0
        topic["family"] = split_df["family"].iloc[0]
        topic["split"] = split_name
        topic["group_type"] = "topic"
        rows_topic.extend(topic.to_dict(orient="records"))

        for group_type, group_df in [("country", country), ("topic", topic)]:
            deltas = group_df["delta_pp"].to_numpy(dtype=float)
            rows_summary.append(
                {
                    "family": split_df["family"].iloc[0],
                    "split": split_name,
                    "group_type": group_type,
                    "mean_delta_pp": float(np.mean(deltas)),
                    "median_delta_pp": float(np.median(deltas)),
                    "n_groups": int(len(deltas)),
                    "n_improved": int(np.sum(deltas > 0.0)),
                }
            )
    return pd.DataFrame(rows_country), pd.DataFrame(rows_topic), pd.DataFrame(rows_summary)


def margin_summary(family_df: pd.DataFrame, validated_keys: set) -> pd.DataFrame:
    rows: List[Dict] = []
    for split_name in ["ambiguous", "validated_ambiguous"]:
        split_df = family_df.loc[subset_mask(family_df, split_name, validated_keys)].copy()
        if split_df.empty:
            continue
        for variant in VARIANT_ORDER:
            variant_df = split_df.loc[split_df["variant"] == variant].copy()
            if variant_df.empty:
                continue
            correct_only = variant_df.loc[variant_df["is_correct"] >= 0.999]
            rows.append(
                {
                    "family": split_df["family"].iloc[0],
                    "split": split_name,
                    "variant": variant,
                    "mean_top_margin": float(variant_df["top_margin"].dropna().mean()),
                    "mean_correct_margin": float(variant_df["correct_margin"].dropna().mean()),
                    "mean_top_margin_correct_only": float(correct_only["top_margin"].dropna().mean()),
                    "mean_correct_margin_correct_only": float(correct_only["correct_margin"].dropna().mean()),
                    "n_items": int(len(variant_df)),
                    "n_correct_items": int(len(correct_only)),
                }
            )
    return pd.DataFrame(rows)


def plot_distribution(df: pd.DataFrame, group_type: str, output_path: Path) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, family in zip(axes, ["1B", "3B"]):
        sub = df.loc[(df["family"] == family) & (df["split"] == "ambiguous")].copy()
        if sub.empty:
            ax.set_axis_off()
            continue
        values = sub["delta_pp"].to_numpy(dtype=float)
        ax.axvline(0.0, color="0.45", linestyle=":", linewidth=1.0)
        ax.scatter(values, np.zeros_like(values), color="#1f77b4", alpha=0.8, s=28)
        ax.set_title(f"{family} ambiguous")
        ax.set_xlabel(r"$\Delta$ accuracy (pp), $(T+, I+) - (T-, I-)$")
        ax.set_yticks([])
        ax.text(
            0.02,
            0.92,
            f"mean={np.mean(values):.2f}\nmedian={np.median(values):.2f}\n>0: {int(np.sum(values > 0))}/{len(values)}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9),
        )
    fig.suptitle(f"LocalNewsQA ambiguous {group_type} deltas")
    fig.tight_layout()
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    validated_keys = load_validated_keys(args.validation_csv)
    family_frames = [
        load_family_rows(args.root_1b, "1B"),
        load_family_rows(args.root_3b, "3B"),
    ]
    full_df = pd.concat(family_frames, ignore_index=True)
    full_df.to_parquet(args.output_root / "item_level_metrics.parquet", index=False)

    accuracy_df = pd.concat(
        [
            accuracy_summary(df, validated_keys, args.bootstrap_samples, args.bootstrap_seed)
            for df in family_frames
        ],
        ignore_index=True,
    )
    accuracy_df.to_csv(args.output_root / "accuracy_with_bootstrap_ci.csv", index=False)

    paired_df = pd.concat(
        [
            paired_comparisons(df, validated_keys, args.bootstrap_samples, args.bootstrap_seed)
            for df in family_frames
        ],
        ignore_index=True,
    )
    paired_df.to_csv(args.output_root / "paired_bootstrap_comparisons.csv", index=False)

    country_df_list = []
    topic_df_list = []
    consistency_summary_list = []
    for df in family_frames:
        country_df, topic_df, summary_df = consistency_tables(df, validated_keys)
        country_df_list.append(country_df)
        topic_df_list.append(topic_df)
        consistency_summary_list.append(summary_df)
    country_df = pd.concat(country_df_list, ignore_index=True) if country_df_list else pd.DataFrame()
    topic_df = pd.concat(topic_df_list, ignore_index=True) if topic_df_list else pd.DataFrame()
    consistency_summary_df = (
        pd.concat(consistency_summary_list, ignore_index=True) if consistency_summary_list else pd.DataFrame()
    )
    country_df.to_csv(args.output_root / "country_delta_distribution.csv", index=False)
    topic_df.to_csv(args.output_root / "topic_delta_distribution.csv", index=False)
    consistency_summary_df.to_csv(args.output_root / "consistency_summary.csv", index=False)

    margin_df = pd.concat(
        [margin_summary(df, validated_keys) for df in family_frames],
        ignore_index=True,
    )
    margin_df.to_csv(args.output_root / "margin_summary.csv", index=False)

    plot_distribution(country_df, "country", args.output_root / "country_delta_distribution.pdf")
    plot_distribution(topic_df, "topic", args.output_root / "topic_delta_distribution.pdf")

    summary = {
        "validated_ambiguous_items": int(len(validated_keys)),
        "accuracy_csv": str(args.output_root / "accuracy_with_bootstrap_ci.csv"),
        "paired_csv": str(args.output_root / "paired_bootstrap_comparisons.csv"),
        "country_csv": str(args.output_root / "country_delta_distribution.csv"),
        "topic_csv": str(args.output_root / "topic_delta_distribution.csv"),
        "margin_csv": str(args.output_root / "margin_summary.csv"),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
