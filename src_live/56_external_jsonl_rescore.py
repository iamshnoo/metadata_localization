#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


LOWER_IS_BETTER_BENCHMARKS = {"worldvaluebench"}


def parse_float_grid(raw: str) -> List[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _token_lengths(
    score_sums: Sequence[Optional[float]],
    score_avgs: Sequence[Optional[float]],
) -> List[float]:
    lengths: List[float] = []
    for score_sum, score_avg in zip(score_sums, score_avgs):
        if score_avg in (None, 0):
            lengths.append(1.0)
        else:
            lengths.append(abs(float(score_sum) / float(score_avg)))
    return lengths


def _selected_scores(record: Dict[str, object], alpha: float, beta: float) -> List[float]:
    primary_sums = record.get("option_loglikelihood_sums") or []
    primary_avgs = record.get("option_loglikelihood_avgs") or []
    primary_lengths = _token_lengths(primary_sums, primary_avgs)
    primary_scores = [
        float(score_sum) / max(length, 1.0) ** alpha
        for score_sum, length in zip(primary_sums, primary_lengths)
    ]

    calibration_sums = record.get("null_calibration_option_loglikelihood_sums")
    calibration_avgs = record.get("null_calibration_option_loglikelihood_avgs")
    if calibration_sums is None or calibration_avgs is None:
        return primary_scores

    calibration_lengths = _token_lengths(calibration_sums, calibration_avgs)
    calibration_scores = [
        float(score_sum) / max(length, 1.0) ** alpha
        for score_sum, length in zip(calibration_sums, calibration_lengths)
    ]
    return [
        primary - beta * calibration
        for primary, calibration in zip(primary_scores, calibration_scores)
    ]


def _predict_answer(record: Dict[str, object], alpha: float, beta: float) -> Optional[str]:
    scores = _selected_scores(record, alpha=alpha, beta=beta)
    if not scores:
        return None
    prompt_options = record.get("prompt_options") or record.get("options") or []
    if not prompt_options:
        return None
    best_idx = int(np.argmax(scores))
    if best_idx >= len(prompt_options):
        return None
    return str(prompt_options[best_idx])


def _wasserstein_equal_weight(values_a: List[float], values_b: List[float]) -> float:
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


def load_records(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def metric_for_records(
    records: Sequence[Dict[str, object]],
    benchmark: str,
    alpha: float,
    beta: float,
) -> Tuple[float, int]:
    if benchmark == "worldvaluebench":
        grouped: Dict[Tuple[str, str, str], Dict[str, List[float]]] = {}
        for record in records:
            pred = _predict_answer(record, alpha=alpha, beta=beta)
            gold = record.get("correct_answer")
            qkey = record.get("question_key")
            continent = record.get("continent")
            country = record.get("country")
            options = record.get("options") or []
            if pred is None or qkey is None:
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
        overall_emd = weighted_sum / total_n if total_n else float("nan")
        return overall_emd, total_n

    correct = 0
    total = 0
    for record in records:
        pred = _predict_answer(record, alpha=alpha, beta=beta)
        if pred is None:
            continue
        total += 1
        correct += int(pred == record.get("correct_answer"))
    accuracy = correct / total if total else float("nan")
    return accuracy, total


def file_uses_calibration(records: Sequence[Dict[str, object]]) -> bool:
    if not records:
        return False
    return records[0].get("null_calibration_option_loglikelihood_sums") is not None


def render_metric(metric: float, benchmark: str) -> str:
    precision = 6 if benchmark != "worldvaluebench" else 6
    return f"{metric:.{precision}f}"


def emit_single(
    records: Sequence[Dict[str, object]],
    benchmark: str,
    label: str,
    alphas: Sequence[float],
    betas: Sequence[float],
) -> None:
    effective_betas = list(betas) if file_uses_calibration(records) else [0.0]
    print(f"[{label}]")
    for alpha in alphas:
        for beta in effective_betas:
            metric, rows = metric_for_records(records, benchmark=benchmark, alpha=alpha, beta=beta)
            print(
                f"  alpha={alpha:.4f} beta={beta:.4f} "
                f"metric={render_metric(metric, benchmark)} rows={rows}"
            )


def emit_pair(
    plus_records: Sequence[Dict[str, object]],
    minus_records: Sequence[Dict[str, object]],
    benchmark: str,
    alphas: Sequence[float],
    betas: Sequence[float],
    limit: int,
) -> None:
    uses_calibration = file_uses_calibration(plus_records) or file_uses_calibration(minus_records)
    effective_betas = list(betas) if uses_calibration else [0.0]
    lower_is_better = benchmark in LOWER_IS_BETTER_BENCHMARKS

    rows = []
    for alpha in alphas:
        for beta in effective_betas:
            plus_metric, plus_n = metric_for_records(
                plus_records, benchmark=benchmark, alpha=alpha, beta=beta
            )
            minus_metric, minus_n = metric_for_records(
                minus_records, benchmark=benchmark, alpha=alpha, beta=beta
            )
            if lower_is_better:
                gap = minus_metric - plus_metric
                better = plus_metric < minus_metric
            else:
                gap = plus_metric - minus_metric
                better = plus_metric > minus_metric
            rows.append(
                {
                    "alpha": alpha,
                    "beta": beta,
                    "plus_metric": plus_metric,
                    "minus_metric": minus_metric,
                    "gap": gap,
                    "better": better,
                    "plus_n": plus_n,
                    "minus_n": minus_n,
                }
            )

    rows.sort(
        key=lambda row: (
            int(row["better"]),
            row["gap"],
            -row["alpha"],
            -row["beta"],
        ),
        reverse=True,
    )
    print("[paired]")
    for row in rows[:limit]:
        print(
            f"  alpha={row['alpha']:.4f} beta={row['beta']:.4f} "
            f"plus={render_metric(row['plus_metric'], benchmark)} "
            f"minus={render_metric(row['minus_metric'], benchmark)} "
            f"gap={row['gap']:.6f} better={str(row['better']).lower()} "
            f"rows_plus={row['plus_n']} rows_minus={row['minus_n']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline alpha/beta rescoring for saved external benchmark JSONL outputs."
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["geomlama", "globalopinionqa", "worldvaluebench", "mmlu"],
    )
    parser.add_argument("--jsonl", type=Path, help="Single JSONL to rescore.")
    parser.add_argument(
        "--plus-jsonl",
        type=Path,
        help="T+/I+ JSONL for paired comparison.",
    )
    parser.add_argument(
        "--minus-jsonl",
        type=Path,
        help="T-/I- JSONL for paired comparison.",
    )
    parser.add_argument(
        "--alphas",
        default="0,0.1,0.25,0.5,0.75,1.0,1.25",
        help="Comma-separated alpha grid.",
    )
    parser.add_argument(
        "--betas",
        default="0,0.25,0.5,0.75,1.0",
        help="Comma-separated beta grid.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum paired rows to print after sorting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alphas = parse_float_grid(args.alphas)
    betas = parse_float_grid(args.betas)

    if args.jsonl is not None and (args.plus_jsonl or args.minus_jsonl):
        raise ValueError("Use either --jsonl or the paired --plus-jsonl/--minus-jsonl inputs.")

    if args.jsonl is None and (args.plus_jsonl is None or args.minus_jsonl is None):
        raise ValueError("Provide either --jsonl or both --plus-jsonl and --minus-jsonl.")

    if args.jsonl is not None:
        emit_single(
            records=load_records(args.jsonl),
            benchmark=args.benchmark,
            label=str(args.jsonl),
            alphas=alphas,
            betas=betas,
        )
        return

    emit_pair(
        plus_records=load_records(args.plus_jsonl),
        minus_records=load_records(args.minus_jsonl),
        benchmark=args.benchmark,
        alphas=alphas,
        betas=betas,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
