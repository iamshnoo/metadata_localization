#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_metric(summary_csv: Path) -> float:
    df = pd.read_csv(summary_csv)
    if "accuracy" in df.columns:
        return float(df["accuracy"].iloc[0])
    if "overall_emd" in df.columns:
        return float(df["overall_emd"].iloc[0])
    raise ValueError(f"Unsupported summary schema: {summary_csv}")


def is_lower_better(benchmark: str) -> bool:
    return benchmark == "worldvaluebench"


def summarize(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for config_dir in sorted(root.iterdir()):
        if not config_dir.is_dir():
            continue
        seed_dir = config_dir / "seed_41"
        if not seed_dir.exists():
            continue
        for bench_dir in sorted(seed_dir.iterdir()):
            if not bench_dir.is_dir():
                continue
            metrics = {}
            for variant_dir in sorted(bench_dir.iterdir()):
                summaries = list(variant_dir.glob("*_eval_summary.csv"))
                if not summaries:
                    continue
                metrics[variant_dir.name] = load_metric(summaries[0])
            if "custom_tplus_eplus" not in metrics or "custom_tminus_eminus" not in metrics:
                continue
            lower_better = is_lower_better(bench_dir.name)
            gap = (
                metrics["custom_tminus_eminus"] - metrics["custom_tplus_eplus"]
                if lower_better
                else metrics["custom_tplus_eplus"] - metrics["custom_tminus_eminus"]
            )
            rows.append(
                {
                    "config": config_dir.name,
                    "benchmark": bench_dir.name,
                    "lower_better": lower_better,
                    "tplus_eplus": metrics["custom_tplus_eplus"],
                    "tminus_eminus": metrics["custom_tminus_eminus"],
                    "delta_favor_tplus_eplus": gap,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/path/to/metacul/results/external_benchmarks_raw3b_tplus_screen",
    )
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    df = summarize(Path(args.root))
    if df.empty:
        print("No completed results found.")
        return

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)

    for bench, sub in df.groupby("benchmark", sort=True):
        print(f"\n## {bench}")
        print(sub.sort_values("delta_favor_tplus_eplus", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
