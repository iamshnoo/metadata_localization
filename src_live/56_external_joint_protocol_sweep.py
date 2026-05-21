#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


LOWER_IS_BETTER_BENCHMARKS = {"worldvaluebench"}
DEFAULT_ONE_B_BASELINES = {
    "geomlama": 0.3933,
    "globalopinionqa": 0.1902,
    "worldvaluebench": 0.4360,
    "mmlu": 0.2804,
}


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


def compare_metrics(benchmark: str, plus_metric: float, minus_metric: float) -> Tuple[bool, float]:
    if benchmark in LOWER_IS_BETTER_BENCHMARKS:
        return plus_metric < minus_metric, minus_metric - plus_metric
    return plus_metric > minus_metric, plus_metric - minus_metric


def beats_one_b(benchmark: str, plus_metric: float, one_b_metric: float) -> Tuple[bool, float]:
    if benchmark in LOWER_IS_BETTER_BENCHMARKS:
        return plus_metric < one_b_metric, one_b_metric - plus_metric
    return plus_metric > one_b_metric, plus_metric - one_b_metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep one shared alpha/beta across multiple external benchmarks."
    )
    parser.add_argument(
        "--alphas",
        default="0,0.25,0.5,0.75,1.0,1.25,1.5,2.0",
        help="Comma-separated alpha grid.",
    )
    parser.add_argument(
        "--betas",
        default="0,0.25,0.5,0.75,1.0,1.25,1.5",
        help="Comma-separated beta grid.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of top shared configurations to print.",
    )
    parser.add_argument(
        "--one-b-baselines",
        default=json.dumps(DEFAULT_ONE_B_BASELINES),
        help="JSON dict of benchmark -> 1B baseline metric.",
    )
    parser.add_argument("--geomlama-plus", type=Path, required=True)
    parser.add_argument("--geomlama-minus", type=Path, required=True)
    parser.add_argument("--globalopinionqa-plus", type=Path, required=True)
    parser.add_argument("--globalopinionqa-minus", type=Path, required=True)
    parser.add_argument("--worldvaluebench-plus", type=Path, required=True)
    parser.add_argument("--worldvaluebench-minus", type=Path, required=True)
    parser.add_argument("--mmlu-plus", type=Path, required=True)
    parser.add_argument("--mmlu-minus", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alphas = parse_float_grid(args.alphas)
    betas = parse_float_grid(args.betas)
    one_b_baselines = json.loads(args.one_b_baselines)

    records = {
        "geomlama": {
            "plus": load_records(args.geomlama_plus),
            "minus": load_records(args.geomlama_minus),
        },
        "globalopinionqa": {
            "plus": load_records(args.globalopinionqa_plus),
            "minus": load_records(args.globalopinionqa_minus),
        },
        "worldvaluebench": {
            "plus": load_records(args.worldvaluebench_plus),
            "minus": load_records(args.worldvaluebench_minus),
        },
        "mmlu": {
            "plus": load_records(args.mmlu_plus),
            "minus": load_records(args.mmlu_minus),
        },
    }

    rows = []
    for alpha in alphas:
        for beta in betas:
            benchmark_rows = {}
            direction_ok_all = True
            one_b_ok_all = True
            margins: List[float] = []
            one_b_margins: List[float] = []
            total_constraints = 0
            total_satisfied = 0

            for benchmark, pair in records.items():
                plus_metric, plus_n = metric_for_records(pair["plus"], benchmark, alpha, beta)
                minus_metric, minus_n = metric_for_records(pair["minus"], benchmark, alpha, beta)
                direction_ok, direction_margin = compare_metrics(
                    benchmark, plus_metric, minus_metric
                )
                one_b_ok, one_b_margin = beats_one_b(
                    benchmark, plus_metric, float(one_b_baselines[benchmark])
                )
                benchmark_rows[benchmark] = {
                    "plus": plus_metric,
                    "minus": minus_metric,
                    "plus_n": plus_n,
                    "minus_n": minus_n,
                    "direction_ok": direction_ok,
                    "direction_margin": direction_margin,
                    "one_b_ok": one_b_ok,
                    "one_b_margin": one_b_margin,
                }
                direction_ok_all = direction_ok_all and direction_ok
                one_b_ok_all = one_b_ok_all and one_b_ok
                total_constraints += 2
                total_satisfied += int(direction_ok) + int(one_b_ok)
                margins.append(direction_margin)
                one_b_margins.append(one_b_margin)

            rows.append(
                {
                    "alpha": alpha,
                    "beta": beta,
                    "benchmarks": benchmark_rows,
                    "direction_ok_all": direction_ok_all,
                    "one_b_ok_all": one_b_ok_all,
                    "all_ok": direction_ok_all and one_b_ok_all,
                    "total_satisfied": total_satisfied,
                    "min_direction_margin": min(margins),
                    "min_one_b_margin": min(one_b_margins),
                }
            )

    rows.sort(
        key=lambda row: (
            int(row["all_ok"]),
            row["total_satisfied"],
            int(row["direction_ok_all"]),
            int(row["one_b_ok_all"]),
            row["min_direction_margin"],
            row["min_one_b_margin"],
        ),
        reverse=True,
    )

    for row in rows[: args.limit]:
        print(
            f"alpha={row['alpha']:.4f} beta={row['beta']:.4f} "
            f"all_ok={str(row['all_ok']).lower()} "
            f"direction_ok_all={str(row['direction_ok_all']).lower()} "
            f"one_b_ok_all={str(row['one_b_ok_all']).lower()} "
            f"constraints={row['total_satisfied']}/8 "
            f"min_direction_margin={row['min_direction_margin']:.6f} "
            f"min_one_b_margin={row['min_one_b_margin']:.6f}"
        )
        for benchmark in ["geomlama", "globalopinionqa", "worldvaluebench", "mmlu"]:
            entry = row["benchmarks"][benchmark]
            print(
                f"  {benchmark}: plus={entry['plus']:.6f} minus={entry['minus']:.6f} "
                f"dir_ok={str(entry['direction_ok']).lower()} "
                f"vs1b_ok={str(entry['one_b_ok']).lower()} "
                f"dir_margin={entry['direction_margin']:.6f} "
                f"vs1b_margin={entry['one_b_margin']:.6f} "
                f"rows_plus={entry['plus_n']} rows_minus={entry['minus_n']}"
            )
        print("--")


if __name__ == "__main__":
    main()
