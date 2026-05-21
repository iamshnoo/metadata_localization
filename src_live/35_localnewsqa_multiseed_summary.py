#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


DEFAULT_PRETRAINED_ROOT = Path(
    "/path/to/metacul/results/downstream_localnewsqa_pretrained_multiseed"
)
DEFAULT_BASELINE_ROOT = Path(
    "/path/to/metacul/results/downstream_localnewsqa_external_baselines_multiseed"
)
DEFAULT_PLOT_CSV = Path(
    "/path/to/metacul/results/plots/plot8/plot_8_pretrained_target_split_multiseed.csv"
)
DEFAULT_TABLE7_LONG_CSV = Path(
    "/path/to/metacul/results/plots/plot8/table7_external_localnewsqa_multiseed_long.csv"
)
DEFAULT_TABLE7_WIDE_CSV = Path(
    "/path/to/metacul/results/plots/plot8/table7_external_localnewsqa_multiseed_wide.csv"
)

PRETRAINED_VARIANTS = [
    {
        "family": "1B",
        "series": "1B T+/I+",
        "run_tag": "1b_tplus_eplus",
        "eval_meta_tag": "with_metadata",
    },
    {
        "family": "1B",
        "series": "1B T+/I-",
        "run_tag": "1b_tplus_eminus",
        "eval_meta_tag": "without_metadata",
    },
    {
        "family": "1B",
        "series": "1B T-/I+",
        "run_tag": "1b_tminus_eplus",
        "eval_meta_tag": "with_metadata",
    },
    {
        "family": "1B",
        "series": "1B T-/I-",
        "run_tag": "1b_tminus_eminus",
        "eval_meta_tag": "without_metadata",
    },
    {
        "family": "3B",
        "series": "3B T+/I+",
        "run_tag": "3b_tplus_eplus",
        "eval_meta_tag": "with_metadata",
    },
    {
        "family": "3B",
        "series": "3B T+/I-",
        "run_tag": "3b_tplus_eminus",
        "eval_meta_tag": "without_metadata",
    },
    {
        "family": "3B",
        "series": "3B T-/I+",
        "run_tag": "3b_tminus_eplus",
        "eval_meta_tag": "with_metadata",
    },
    {
        "family": "3B",
        "series": "3B T-/I-",
        "run_tag": "3b_tminus_eminus",
        "eval_meta_tag": "without_metadata",
    },
]

TABLE7_MODELS = [
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

META_TAGS = {
    "with_metadata": "With metadata",
    "without_metadata": "Without metadata",
}

SPLIT_ORDER = {"Overall": 0, "Explicit": 1, "Ambiguous": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate LocalNewsQA shuffled option-text results across multiple seeds."
    )
    parser.add_argument(
        "--pretrained-root",
        type=Path,
        default=DEFAULT_PRETRAINED_ROOT,
        help="Root with per-seed pretrained LocalNewsQA runs for Figure 9.",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=DEFAULT_BASELINE_ROOT,
        help="Root with per-seed external baseline LocalNewsQA runs for Table 7.",
    )
    parser.add_argument(
        "--seeds",
        default="41,42,43,44,45",
        help="Comma-separated seed list used for option shuffling.",
    )
    parser.add_argument("--plot-csv", type=Path, default=DEFAULT_PLOT_CSV)
    parser.add_argument("--table7-long-csv", type=Path, default=DEFAULT_TABLE7_LONG_CSV)
    parser.add_argument("--table7-wide-csv", type=Path, default=DEFAULT_TABLE7_WIDE_CSV)
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


def aggregate_pretrained(args: argparse.Namespace, seeds: List[int]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for variant in PRETRAINED_VARIANTS:
        per_split_values: Dict[str, List[float]] = {
            "Overall": [],
            "Explicit": [],
            "Ambiguous": [],
        }
        total_rows = {"Overall": None, "Explicit": None, "Ambiguous": None}
        for seed in seeds:
            seed_dir = args.pretrained_root / f"seed_{seed}"
            path = find_single(
                f"localnewsqa_eval_target_{variant['eval_meta_tag']}_custom*{variant['run_tag']}*_c0*.jsonl",
                seed_dir,
            )
            if path is None:
                print(f"[missing] Figure 9 seed={seed} variant={variant['run_tag']}")
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


def aggregate_table7(args: argparse.Namespace, seeds: List[int]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model_slug, label in TABLE7_MODELS:
        for meta_tag in ("with_metadata", "without_metadata"):
            per_split_values: Dict[str, List[float]] = {
                "Overall": [],
                "Explicit": [],
                "Ambiguous": [],
            }
            total_rows = {"Overall": None, "Explicit": None, "Ambiguous": None}
            for seed in seeds:
                seed_dir = args.baseline_root / f"seed_{seed}"
                path = find_single(
                    f"localnewsqa_{model_slug}_{meta_tag}_target*.jsonl",
                    seed_dir,
                )
                if path is None:
                    print(f"[missing] Table 7 seed={seed} model={model_slug} meta={meta_tag}")
                    continue
                metrics = load_localnewsqa_metrics(path)
                for split in ("Overall", "Explicit", "Ambiguous"):
                    per_split_values[split].append(metrics[f"{split}_accuracy"])
                    total_rows[split] = int(metrics[f"{split}_total"])

            for split in ("Overall", "Explicit", "Ambiguous"):
                values = per_split_values[split]
                if not values:
                    continue
                rows.append(
                    {
                        "model_slug": model_slug,
                        "model_label": label,
                        "meta_tag": meta_tag,
                        "meta_label": META_TAGS[meta_tag],
                        "split": split,
                        "accuracy": mean(values),
                        "accuracy_std": sample_std(values),
                        "seed_count": len(values),
                        "total": total_rows[split],
                    }
                )
    rows.sort(key=lambda row: (row["model_label"], row["meta_tag"], SPLIT_ORDER[row["split"]]))
    return rows


def build_table7_wide(long_rows: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(long_rows)
    if df.empty:
        return pd.DataFrame()

    acc = (
        df.pivot_table(
            index=["model_slug", "model_label"],
            columns=["meta_tag", "split"],
            values="accuracy",
            aggfunc="first",
        )
        .sort_index(axis=1)
        .reset_index()
    )
    std = (
        df.pivot_table(
            index=["model_slug", "model_label"],
            columns=["meta_tag", "split"],
            values="accuracy_std",
            aggfunc="first",
        )
        .sort_index(axis=1)
        .reset_index()
    )
    acc.columns = [
        "_".join([part for part in col if part]).strip("_") if isinstance(col, tuple) else col
        for col in acc.columns
    ]
    std.columns = [
        (
            "_".join([part for part in col if part]).strip("_") + "_std"
            if isinstance(col, tuple) and any(col)
            else col
        )
        for col in std.columns
    ]
    for df_part in (acc, std):
        rename_map = {}
        for col in df_part.columns:
            if col.startswith("model_slug"):
                rename_map[col] = "model_slug"
            elif col.startswith("model_label"):
                rename_map[col] = "model_label"
        if rename_map:
            df_part.rename(columns=rename_map, inplace=True)
    merged = acc.merge(std, on=["model_slug", "model_label"], how="left")
    return merged


def main() -> int:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)

    plot_rows = aggregate_pretrained(args, seeds)
    table7_rows = aggregate_table7(args, seeds)

    if plot_rows:
        ensure_parent(args.plot_csv)
        pd.DataFrame(plot_rows).to_csv(args.plot_csv, index=False)
        print(f"[ok] Wrote Figure 9 CSV: {args.plot_csv}")
    else:
        print("[warn] No completed Figure 9 rows found.")

    if table7_rows:
        ensure_parent(args.table7_long_csv)
        pd.DataFrame(table7_rows).to_csv(args.table7_long_csv, index=False)
        print(f"[ok] Wrote Table 7 long CSV: {args.table7_long_csv}")

        table7_wide = build_table7_wide(table7_rows)
        ensure_parent(args.table7_wide_csv)
        table7_wide.to_csv(args.table7_wide_csv, index=False)
        print(f"[ok] Wrote Table 7 wide CSV: {args.table7_wide_csv}")
    else:
        print("[warn] No completed Table 7 rows found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
