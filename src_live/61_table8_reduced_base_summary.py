#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


DEFAULT_RESULTS_ROOT = Path(
    "/path/to/metacul/results/final_benchmark_matrix/closed_form"
)
DEFAULT_LONG_CSV = Path(
    "/path/to/metacul/results/plots/plot8/table8_external_eval_baseonly_current_long.csv"
)
DEFAULT_WIDE_CSV = Path(
    "/path/to/metacul/results/plots/plot8/table8_external_eval_baseonly_current_wide.csv"
)
DEFAULT_LATEX_ROWS = Path(
    "/path/to/metacul/results/plots/plot8/table8_external_eval_baseonly_current_rows.tex"
)

DATASETS: List[Tuple[str, str, str]] = [
    ("blend", "BLEnD", "accuracy"),
    ("globalmmlu_cs", "Global-MMLU CS", "accuracy"),
    ("normad", "NormAd", "accuracy"),
]

MODEL_GROUPS = [
    (
        "MAPLE 1B",
        "maple_1b",
        [
            ("custom_tplus_eplus", "(T+, I+)"),
            ("custom_tplus_eminus", "(T+, I-)"),
            ("custom_tminus_eplus", "(T-, I+)"),
            ("custom_tminus_eminus", "(T-, I-)"),
        ],
    ),
    (
        "MAPLE 3B",
        "maple_3b",
        [
            ("custom_tplus_eplus", "(T+, I+)"),
            ("custom_tplus_eminus", "(T+, I-)"),
            ("custom_tminus_eplus", "(T-, I+)"),
            ("custom_tminus_eminus", "(T-, I-)"),
        ],
    ),
    (
        "LLaMA-3.2-1B",
        "llama32_1b",
        [
            ("llama3_chat_with_metadata", "(I+)"),
            ("llama3_chat_without_metadata", "(I-)"),
        ],
    ),
    (
        "LLaMA-3.2-3B",
        "llama32_3b",
        [
            ("llama3_chat_with_metadata", "(I+)"),
            ("llama3_chat_without_metadata", "(I-)"),
        ],
    ),
    (
        "Gemma-4-E2B",
        "gemma4_e2b",
        [
            ("llama3_chat_with_metadata", "(I+)"),
            ("llama3_chat_without_metadata", "(I-)"),
        ],
    ),
    (
        "Gemma-4-E4B",
        "gemma4_e4b",
        [
            ("llama3_chat_with_metadata", "(I+)"),
            ("llama3_chat_without_metadata", "(I-)"),
        ],
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate the harmonized reduced base-only Table 8 external benchmark results."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--seeds", default="41,42,43,44,45")
    parser.add_argument("--long-csv", type=Path, default=DEFAULT_LONG_CSV)
    parser.add_argument("--wide-csv", type=Path, default=DEFAULT_WIDE_CSV)
    parser.add_argument("--latex-rows", type=Path, default=DEFAULT_LATEX_ROWS)
    return parser.parse_args()


def parse_seed_list(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def mean(values: Iterable[float]) -> float:
    vals = list(values)
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
    return float(row.iloc[0]["accuracy"]) * 100.0


def read_emd(summary_json: Path) -> float:
    with summary_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return float(payload["overall_emd"])


def collect_rows(results_root: Path, seeds: List[int]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    missing: List[str] = []

    for model_label, model_slug, variant_specs in MODEL_GROUPS:
        model_root = results_root / model_slug / "base"
        for dataset_slug, dataset_label, metric_name in DATASETS:
            for variant_name, column_label in variant_specs:
                values: List[float] = []
                for seed in seeds:
                    variant_dir = model_root / f"seed_{seed}" / dataset_slug / variant_name
                    if metric_name == "accuracy":
                        summary_path = variant_dir / f"{dataset_slug}_eval_summary.csv"
                        if not summary_path.exists():
                            missing.append(str(summary_path))
                            continue
                        values.append(read_accuracy(summary_path, variant_name))
                    else:
                        summary_path = variant_dir / f"{dataset_slug}_{variant_name}_emd_summary.json"
                        if not summary_path.exists():
                            missing.append(str(summary_path))
                            continue
                        values.append(read_emd(summary_path))
                if len(values) != len(seeds):
                    raise RuntimeError(
                        f"Incomplete coverage for {model_label} {dataset_label} {variant_name}: "
                        f"got {len(values)} of {len(seeds)} seeds"
                    )
                rows.append(
                    {
                        "dataset": dataset_label,
                        "benchmark": dataset_slug,
                        "metric": metric_name,
                        "model_group": model_label,
                        "variant": variant_name,
                        "column_label": column_label,
                        "value": mean(values),
                        "value_std": sample_std(values),
                        "seed_count": len(values),
                    }
                )

    if missing:
        preview = "\n".join(missing[:20])
        raise RuntimeError(f"Missing required summary files.\n{preview}")
    return rows


def build_wide_table(rows: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
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


def fmt_value(metric: str, value: float) -> str:
    return f"{value:.3f}" if metric == "emd" else f"{value:.2f}"


def format_cell(metric: str, value: float, best: bool, positive: bool) -> str:
    macro = "tableprimaryplus" if positive else "tableprimaryminus"
    payload = fmt_value(metric, value)
    if best:
        payload = f"\\textbf{{{payload}}}"
    return f"\\{macro}{{{payload}}}"


def build_latex_rows(rows: List[Dict[str, object]]) -> str:
    df = pd.DataFrame(rows)
    order = [dataset_label for _, dataset_label, _ in DATASETS]
    metric_lookup = {dataset_label: metric_name for _, dataset_label, metric_name in DATASETS}

    lines: List[str] = []
    lines.append(r"\multicolumn{17}{l}{\textit{External datasets - Metric: Accuracy (\%)}} \\")
    for dataset_label in order:
        metric_name = metric_lookup[dataset_label]
        if metric_name != "accuracy":
            continue
        pieces = [dataset_label]
        for model_label, _, variant_specs in MODEL_GROUPS:
            block = df[(df["dataset"] == dataset_label) & (df["model_group"] == model_label)]
            values = [float(block[block["variant"] == variant_name]["value"].iloc[0]) for variant_name, _ in variant_specs]
            best_value = max(values)
            for idx, (variant_name, _) in enumerate(variant_specs):
                value = values[idx]
                pieces.append(
                    format_cell(
                        metric_name,
                        value,
                        math.isclose(value, best_value, rel_tol=0.0, abs_tol=1e-12),
                        idx == 0,
                    )
                )
        lines.append(" & ".join(pieces) + r" \\")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{17}{l}{\textit{External dataset - Metric: Earth Mover Distance}} \\")
    for dataset_label in order:
        metric_name = metric_lookup[dataset_label]
        if metric_name != "emd":
            continue
        pieces = [dataset_label]
        for model_label, _, variant_specs in MODEL_GROUPS:
            block = df[(df["dataset"] == dataset_label) & (df["model_group"] == model_label)]
            values = [float(block[block["variant"] == variant_name]["value"].iloc[0]) for variant_name, _ in variant_specs]
            best_value = min(values)
            for idx, (variant_name, _) in enumerate(variant_specs):
                value = values[idx]
                pieces.append(
                    format_cell(
                        metric_name,
                        value,
                        math.isclose(value, best_value, rel_tol=0.0, abs_tol=1e-12),
                        idx == 0,
                    )
                )
        lines.append(" & ".join(pieces) + r" \\")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    rows = collect_rows(args.results_root, seeds)

    long_df = pd.DataFrame(rows)
    ensure_parent(args.long_csv)
    long_df.to_csv(args.long_csv, index=False)
    print(f"[ok] Wrote long CSV: {args.long_csv}")

    wide_df = build_wide_table(rows)
    ensure_parent(args.wide_csv)
    wide_df.to_csv(args.wide_csv, index=False)
    print(f"[ok] Wrote wide CSV: {args.wide_csv}")

    latex_rows = build_latex_rows(rows)
    ensure_parent(args.latex_rows)
    args.latex_rows.write_text(latex_rows, encoding="utf-8")
    print(f"[ok] Wrote LaTeX rows: {args.latex_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
