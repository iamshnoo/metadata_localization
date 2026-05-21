#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


DEFAULT_OURS_1B_ROOT = Path(
    "/path/to/metacul/results/external_benchmarks_pretrained_multiseed/ours_1b"
)
DEFAULT_OURS_3B_ROOT = Path(
    "/path/to/metacul/results/external_benchmarks_pretrained_multiseed/ours_3b"
)
DEFAULT_LLAMA_ROOT = Path(
    "/path/to/metacul/results/external_benchmarks_pretrained_multiseed/llama32_1b"
)
DEFAULT_LONG_CSV = Path(
    "/path/to/metacul/results/plots/plot8/table8_external_eval_multiseed_long.csv"
)
DEFAULT_WIDE_CSV = Path(
    "/path/to/metacul/results/plots/plot8/table8_external_eval_multiseed_wide.csv"
)

BENCHMARKS = [
    ("geomlama", "GeoMLaMA", "accuracy"),
    ("globalopinionqa", "GlobalOpinionQA", "accuracy"),
    ("worldvaluebench", "WorldValuesBench", "emd"),
    ("mmlu", "Overall", "accuracy"),
]

VARIANT_GROUPS = [
    (
        "MAPLE 1B",
        DEFAULT_OURS_1B_ROOT,
        [
            ("custom_tplus_eplus", "(T+, I+)"),
            ("custom_tplus_eminus", "(T+, I-)"),
            ("custom_tminus_eplus", "(T-, I+)"),
            ("custom_tminus_eminus", "(T-, I-)"),
        ],
    ),
    (
        "MAPLE 3B",
        DEFAULT_OURS_3B_ROOT,
        [
            ("custom_tplus_eplus", "(T+, I+)"),
            ("custom_tplus_eminus", "(T+, I-)"),
            ("custom_tminus_eplus", "(T-, I+)"),
            ("custom_tminus_eminus", "(T-, I-)"),
        ],
    ),
    (
        "LLaMA-3.2-1B",
        DEFAULT_LLAMA_ROOT,
        [
            ("llama3_chat_with_metadata", "(I+)"),
            ("llama3_chat_without_metadata", "(I-)"),
        ],
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate shuffled multi-seed external benchmark summaries for Table 8."
    )
    parser.add_argument("--ours-1b-root", type=Path, default=DEFAULT_OURS_1B_ROOT)
    parser.add_argument("--ours-3b-root", type=Path, default=DEFAULT_OURS_3B_ROOT)
    parser.add_argument("--llama-root", type=Path, default=DEFAULT_LLAMA_ROOT)
    parser.add_argument(
        "--seeds",
        default="41,42,43,44,45",
        help="Comma-separated seed list used for option shuffling.",
    )
    parser.add_argument("--long-csv", type=Path, default=DEFAULT_LONG_CSV)
    parser.add_argument("--wide-csv", type=Path, default=DEFAULT_WIDE_CSV)
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


def read_accuracy(summary_csv: Path, variant: str) -> float:
    df = pd.read_csv(summary_csv)
    row = df[df["variant"] == variant]
    if row.empty:
        raise ValueError(f"Variant {variant} not found in {summary_csv}")
    status = str(row.iloc[0].get("status", "ok"))
    if status != "ok":
        raise ValueError(f"Variant {variant} has non-ok status={status} in {summary_csv}")
    return float(row.iloc[0]["accuracy"])


def read_emd(summary_json: Path) -> float:
    with summary_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return float(payload["overall_emd"])


def aggregate_rows(args: argparse.Namespace, seeds: List[int]) -> List[Dict[str, object]]:
    root_map = {
        "MAPLE 1B": args.ours_1b_root,
        "MAPLE 3B": args.ours_3b_root,
        "LLaMA-3.2-1B": args.llama_root,
    }
    rows: List[Dict[str, object]] = []

    for model_group, _, variant_specs in VARIANT_GROUPS:
        model_root = root_map[model_group]
        for benchmark_slug, dataset_label, metric_name in BENCHMARKS:
            for variant_name, column_label in variant_specs:
                values: List[float] = []
                for seed in seeds:
                    variant_dir = model_root / f"seed_{seed}" / benchmark_slug / variant_name
                    if metric_name == "accuracy":
                        summary_path = variant_dir / f"{benchmark_slug}_eval_summary.csv"
                        if not summary_path.exists():
                            print(
                                f"[missing] Table 8 seed={seed} group={model_group} benchmark={benchmark_slug} variant={variant_name}"
                            )
                            continue
                        try:
                            values.append(read_accuracy(summary_path, variant_name))
                        except ValueError as exc:
                            print(
                                f"[skip] Table 8 seed={seed} group={model_group} benchmark={benchmark_slug} variant={variant_name}: {exc}"
                            )
                            continue
                    else:
                        summary_path = variant_dir / f"{benchmark_slug}_{variant_name}_emd_summary.json"
                        if not summary_path.exists():
                            print(
                                f"[missing] Table 8 seed={seed} group={model_group} benchmark={benchmark_slug} variant={variant_name}"
                            )
                            continue
                        values.append(read_emd(summary_path))

                if not values:
                    continue
                rows.append(
                    {
                        "dataset": dataset_label,
                        "benchmark": benchmark_slug,
                        "metric": metric_name,
                        "model_group": model_group,
                        "variant": variant_name,
                        "column_label": column_label,
                        "value": mean(values),
                        "value_std": sample_std(values),
                        "seed_count": len(values),
                    }
                )
    return rows


def build_wide_table(long_rows: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(long_rows)
    if df.empty:
        return pd.DataFrame()

    value_df = (
        df.pivot_table(
            index=["dataset", "metric"],
            columns=["model_group", "column_label"],
            values="value",
            aggfunc="first",
        )
        .sort_index(axis=1)
        .reset_index()
    )
    std_df = (
        df.pivot_table(
            index=["dataset", "metric"],
            columns=["model_group", "column_label"],
            values="value_std",
            aggfunc="first",
        )
        .sort_index(axis=1)
        .reset_index()
    )
    value_df.columns = [
        "_".join([part for part in col if part]).strip("_") if isinstance(col, tuple) else col
        for col in value_df.columns
    ]
    std_df.columns = [
        (
            "_".join([part for part in col if part]).strip("_") + "_std"
            if isinstance(col, tuple) and any(col)
            else col
        )
        for col in std_df.columns
    ]
    for df_part in (value_df, std_df):
        rename_map = {}
        for col in df_part.columns:
            if col.startswith("dataset"):
                rename_map[col] = "dataset"
            elif col.startswith("metric"):
                rename_map[col] = "metric"
        if rename_map:
            df_part.rename(columns=rename_map, inplace=True)
    return value_df.merge(std_df, on=["dataset", "metric"], how="left")


def main() -> int:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    rows = aggregate_rows(args, seeds)
    if not rows:
        print("[warn] No completed Table 8 rows found.")
        return 0

    long_df = pd.DataFrame(rows)
    ensure_parent(args.long_csv)
    long_df.to_csv(args.long_csv, index=False)
    print(f"[ok] Wrote Table 8 long CSV: {args.long_csv}")

    wide_df = build_wide_table(rows)
    ensure_parent(args.wide_csv)
    wide_df.to_csv(args.wide_csv, index=False)
    print(f"[ok] Wrote Table 8 wide CSV: {args.wide_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
