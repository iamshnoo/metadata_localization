#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


ACCURACY_BENCHMARKS = {
    "geomlama",
    "globalopinionqa",
    "globalmmlu_cs",
    "normad",
    "blend",
    "mmlu",
}
EXPECTED_ROWS = {
    "geomlama": 150,
    "globalopinionqa": 10275,
    "worldvaluebench": 1405,
    "globalmmlu_cs": 792,
    "normad": 2633,
    "blend": 393,
    "mmlu": 14042,
}


def _read_metric(path: Path, benchmark: str) -> Optional[Tuple[float, Optional[int], str]]:
    if path.suffix == ".csv":
        row = pd.read_csv(path).iloc[0].to_dict()
        if benchmark in ACCURACY_BENCHMARKS:
            processed = int(row.get("processed_total", row.get("total_rows", 0))) or None
            return float(row["accuracy"]), processed, "summary"
        return None

    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        if benchmark == "worldvaluebench":
            return float(payload["overall_emd"]), int(payload.get("scored_rows", 0)) or None, "summary"
        return None

    return None


def _read_jsonl_metric(path: Path, benchmark: str) -> Optional[Tuple[float, int, str]]:
    total = 0
    value_sum = 0.0
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            total += 1
            if benchmark in ACCURACY_BENCHMARKS:
                value_sum += 1.0 if record.get("is_correct") else 0.0
            elif benchmark == "worldvaluebench":
                if "emd" not in record:
                    return None
                value_sum += float(record["emd"])
            else:
                return None
    if total == 0:
        return None
    return value_sum / total, total, "jsonl"


def _load_variant_metric(
    benchmark_dir: Path, variant: str, benchmark: str
) -> Optional[Tuple[float, Optional[int], str]]:
    if benchmark in ACCURACY_BENCHMARKS:
        summary_candidates = sorted(benchmark_dir.glob(f"{variant}/*summary.csv")) + sorted(
            benchmark_dir.glob(f"{variant}/*summary.json")
        )
    else:
        summary_candidates = sorted(benchmark_dir.glob(f"{variant}/*summary.json"))
    for candidate in summary_candidates:
        metric = _read_metric(candidate, benchmark)
        if metric is not None:
            return metric

    # Packed/survivor runs may write flat files directly under benchmark_dir.
    flat_summary_candidates = []
    if benchmark in ACCURACY_BENCHMARKS:
        flat_summary_candidates.extend(sorted(benchmark_dir.glob(f"*{variant}*summary.csv")))
        flat_summary_candidates.extend(sorted(benchmark_dir.glob(f"*{variant}*summary.json")))
    else:
        flat_summary_candidates.extend(sorted(benchmark_dir.glob(f"*{variant}*summary.json")))
    for candidate in flat_summary_candidates:
        metric = _read_metric(candidate, benchmark)
        if metric is not None:
            return metric

    jsonl_candidates = sorted(benchmark_dir.glob(f"{variant}/*.jsonl"))
    if jsonl_candidates:
        return _read_jsonl_metric(jsonl_candidates[0], benchmark)

    flat_jsonl_candidates = sorted(benchmark_dir.glob(f"*{variant}.jsonl"))
    if flat_jsonl_candidates:
        return _read_jsonl_metric(flat_jsonl_candidates[0], benchmark)

    return None


def collect(root: Path, benchmarks: List[str], variants: List[str]) -> Dict[str, Dict[str, List[float]]]:
    results: Dict[str, Dict[str, List[float]]] = {b: {v: [] for v in variants} for b in benchmarks}
    seed_dirs = sorted([p for p in root.glob("seed_*") if p.is_dir()])
    if not seed_dirs:
        seed_dirs = [root]

    for seed_dir in seed_dirs:
        for benchmark in benchmarks:
            benchmark_dir = seed_dir / benchmark
            if not benchmark_dir.exists():
                continue
            for variant in variants:
                metric = _load_variant_metric(benchmark_dir, variant, benchmark)
                if metric is not None:
                    results[benchmark][variant].append(metric)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize external benchmark protocol roots.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=[
            "geomlama",
            "globalopinionqa",
            "worldvaluebench",
            "globalmmlu_cs",
            "normad",
            "blend",
        ],
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["custom_tplus_eplus", "custom_tminus_eminus"],
    )
    args = parser.parse_args()

    results = collect(args.root, args.benchmarks, args.variants)
    for benchmark in args.benchmarks:
        print(f"[{benchmark}]")
        for variant in args.variants:
            values = results[benchmark][variant]
            if not values:
                print(f"  {variant}: MISSING")
                continue
            metrics = [v[0] for v in values]
            mean = sum(metrics) / len(metrics)
            rendered = []
            for metric, processed, source in values:
                suffix = ""
                expected = EXPECTED_ROWS.get(benchmark)
                if processed is not None:
                    suffix = f" rows={processed}"
                    if expected is not None and processed != expected:
                        suffix += f"/{expected}"
                rendered.append(f"{metric:.6f} ({source}{suffix})")
            joined = ", ".join(rendered)
            print(f"  {variant}: mean={mean:.6f} n={len(metrics)} values=[{joined}]")


if __name__ == "__main__":
    main()
