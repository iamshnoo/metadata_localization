#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BENCH_INFO = {
    "geomlama": {"lower_better": False},
    "globalopinionqa": {"lower_better": False},
    "worldvaluebench": {"lower_better": True},
    "mmlu": {"lower_better": False},
}


def _wasserstein_equal_weight(values_a: list[float], values_b: list[float]) -> float:
    if not values_a or not values_b:
        return float("nan")
    a = np.sort(np.array(values_a, dtype=float))
    b = np.sort(np.array(values_b, dtype=float))
    if len(a) == len(b):
        return float(np.mean(np.abs(a - b)))
    n = max(len(a), len(b))
    q = (np.arange(n) + 0.5) / n
    qa = np.quantile(a, q, method="linear")
    qb = np.quantile(b, q, method="linear")
    return float(np.mean(np.abs(qa - qb)))


def compute_metric(path: Path) -> tuple[int, str, float]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        if "overall_emd" in payload:
            return int(payload.get("scored_rows", 0)), "emd", float(payload["overall_emd"])

    rows = 0
    total = 0.0
    metric = "accuracy"
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            record = json.loads(line)
            if "emd" in record:
                metric = "emd"
                pred = record.get("processed_answer")
                gold = record.get("correct_answer")
                qkey = record.get("question_key")
                continent = record.get("continent")
                country = record.get("country")
                options = record.get("options") or []
                if pred is None or gold is None or qkey is None:
                    continue
                try:
                    pred_i = int(pred)
                    gold_i = int(gold)
                    min_scale = int(min(int(x) for x in options))
                    max_scale = int(max(int(x) for x in options))
                except Exception:
                    continue
                if max_scale <= min_scale:
                    continue
                pred_n = (pred_i - min_scale) / (max_scale - min_scale)
                gold_n = (gold_i - min_scale) / (max_scale - min_scale)
                slot = grouped.setdefault(
                    (str(qkey), str(continent), str(country)),
                    {"correct_answers": [], "predicted_answers": []},
                )
                slot["correct_answers"].append(gold_n)
                slot["predicted_answers"].append(pred_n)
            else:
                total += float(bool(record.get("is_correct")))
    if metric == "emd":
        weighted_sum = 0.0
        total_n = 0
        for values in grouped.values():
            emd = _wasserstein_equal_weight(
                values["correct_answers"],
                values["predicted_answers"],
            )
            n = len(values["predicted_answers"])
            weighted_sum += emd * n
            total_n += n
        value = weighted_sum / total_n if total_n else float("nan")
        return total_n, metric, value

    value = total / rows if rows else 0.0
    if metric == "accuracy":
        value *= 100.0
    return rows, metric, value


def infer_config_and_seed(path: Path) -> tuple[str, str]:
    parts = path.parts
    config = "unknown"
    seed = "unknown"
    for i, part in enumerate(parts):
        if part.startswith("seed_"):
            seed = part
            if i > 0:
                config = parts[i - 1]
            break
    return config, seed


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize fixed-protocol external benchmark sweeps.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Detailed per-file summary CSV.",
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        required=True,
        help="Pivoted plus/minus comparison CSV.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=1,
        help="Minimum rows required for a file to be included.",
    )
    args = parser.parse_args()

    rows = []
    seen_keys = set()
    metric_files = sorted(args.root.glob("**/*_emd_summary.json")) + sorted(args.root.glob("**/*.jsonl"))
    for fp in metric_files:
        parts = fp.parts
        benchmark = next((b for b in BENCH_INFO if b in parts), None)
        if benchmark is None:
            continue
        variant = next(
            (v for v in ("custom_tplus_eplus", "custom_tminus_eminus", "with_metadata", "without_metadata") if v in parts or v in fp.name),
            None,
        )
        if variant is None:
            continue
        config, seed = infer_config_and_seed(fp)
        key = (config, seed, benchmark, variant)
        if key in seen_keys:
            continue
        row_count, metric, value = compute_metric(fp)
        if benchmark == "worldvaluebench" and metric != "emd":
            continue
        if row_count < args.min_rows:
            continue
        rows.append(
            {
                "config": config,
                "seed": seed,
                "benchmark": benchmark,
                "variant": variant,
                "metric": metric,
                "rows": row_count,
                "value": value,
                "path": str(fp),
            }
        )
        seen_keys.add(key)

    if not rows:
        print("[warn] no rows found")
        return 0

    df = pd.DataFrame(rows).sort_values(["config", "seed", "benchmark", "variant"]).reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    compare_rows = []
    for (config, seed, benchmark), group in df.groupby(["config", "seed", "benchmark"]):
        plus = group[group["variant"].isin(["custom_tplus_eplus", "with_metadata"])]
        minus = group[group["variant"].isin(["custom_tminus_eminus", "without_metadata"])]
        if plus.empty or minus.empty:
            continue
        plus_row = plus.iloc[0]
        minus_row = minus.iloc[0]
        plus_val = float(plus_row["value"])
        minus_val = float(minus_row["value"])
        lower_better = BENCH_INFO[benchmark]["lower_better"]
        compare_rows.append(
            {
                "config": config,
                "seed": seed,
                "benchmark": benchmark,
                "metric": str(plus_row["metric"]),
                "plus_rows": int(plus_row["rows"]),
                "minus_rows": int(minus_row["rows"]),
                "plus_value": plus_val,
                "minus_value": minus_val,
                "req1_satisfied": plus_val < minus_val if lower_better else plus_val > minus_val,
            }
        )

    comp_df = pd.DataFrame(compare_rows).sort_values(["config", "seed", "benchmark"]).reset_index(drop=True)
    comp_df.to_csv(args.comparison_csv, index=False)

    print(f"[ok] wrote {args.output_csv}")
    print(f"[ok] wrote {args.comparison_csv}")
    if not comp_df.empty:
        print(comp_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
