#!/usr/bin/env python3
import argparse
import csv
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


RESULT_ROOT = Path("/path/to/metacul/results/external_protocol_probes")
REPORT_ROOT = RESULT_ROOT / "_reports"

DEFAULT_BENCHMARKS = [
    "geomlama",
    "globalopinionqa",
    "worldvaluebench",
    "globalmmlu_cs",
    "normad",
    "blend",
]
LOWER_IS_BETTER = {"worldvaluebench"}


@dataclass(frozen=True)
class PairSpec:
    positive: str
    negative: str
    label: str


PAIR_SPECS: Dict[str, PairSpec] = {
    "maple_1b": PairSpec("custom_tplus_eplus", "custom_tminus_eminus", "T+I+ vs T-I-"),
    "maple_3b": PairSpec("custom_tplus_eplus", "custom_tminus_eminus", "T+I+ vs T-I-"),
    "maple_1b_chat": PairSpec("custom_tplus_eplus", "custom_tminus_eminus", "T+I+ vs T-I-"),
    "maple_3b_chat": PairSpec("custom_tplus_eplus", "custom_tminus_eminus", "T+I+ vs T-I-"),
    "llama32_1b": PairSpec(
        "llama3_chat_with_metadata",
        "llama3_chat_without_metadata",
        "I+ vs I-",
    ),
}


def _read_summary_csv(path: Path, variant: str) -> Optional[Tuple[float, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("variant") != variant:
                continue
            metric = row.get("accuracy")
            processed = row.get("processed_total") or row.get("total_rows") or "0"
            if metric in {None, ""}:
                return None
            return float(metric), int(float(processed))
    return None


def _read_emd_summary_json(path: Path) -> Optional[Tuple[float, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "overall_emd" not in payload:
        return None
    return float(payload["overall_emd"]), int(payload.get("scored_rows", 0) or 0)


def _score_range(row: Dict[str, object]) -> Optional[float]:
    scores = row.get("option_loglikelihood_selected_scores")
    if not isinstance(scores, list) or len(scores) < 2:
        return None
    vals = []
    for score in scores:
        try:
            val = float(score)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(val):
            return None
        vals.append(val)
    return max(vals) - min(vals)


def _read_jsonl(path: Path, benchmark: str) -> Dict[str, object]:
    total = 0
    correct = 0
    emd_sum = 0.0
    emd_total = 0
    score_rows = 0
    constant_score_rows = 0
    tokenizer_sources = set()
    tokenizer_classes = set()
    tokenizer_vocab_sizes = set()
    first_option_predictions = 0
    predicted_rows = 0

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            tokenizer_source = row.get("tokenizer_source")
            if tokenizer_source:
                tokenizer_sources.add(str(tokenizer_source))
            tokenizer_class = row.get("tokenizer_class")
            if tokenizer_class:
                tokenizer_classes.add(str(tokenizer_class))
            tokenizer_vocab_size = row.get("tokenizer_vocab_size")
            if tokenizer_vocab_size is not None:
                tokenizer_vocab_sizes.add(str(tokenizer_vocab_size))

            score_span = _score_range(row)
            if score_span is not None:
                score_rows += 1
                if score_span <= 1e-8:
                    constant_score_rows += 1

            processed_answer = row.get("processed_answer")
            if processed_answer is not None:
                predicted_rows += 1
                prompt_options = row.get("prompt_options") or row.get("options") or []
                if isinstance(prompt_options, list) and prompt_options:
                    if processed_answer == prompt_options[0]:
                        first_option_predictions += 1

            if benchmark == "worldvaluebench" and row.get("emd") is not None:
                emd_sum += float(row["emd"])
                emd_total += 1
            else:
                is_correct = row.get("is_correct")
                if is_correct is None:
                    is_correct = processed_answer == row.get("correct_answer")
                correct += int(bool(is_correct))

    if benchmark == "worldvaluebench" and emd_total:
        metric = emd_sum / emd_total
        metric_rows = emd_total
    else:
        metric = correct / total if total else float("nan")
        metric_rows = total

    return {
        "metric": metric,
        "rows": metric_rows,
        "jsonl_rows": total,
        "score_rows": score_rows,
        "constant_score_rate": constant_score_rows / score_rows if score_rows else float("nan"),
        "first_option_prediction_rate": (
            first_option_predictions / predicted_rows if predicted_rows else float("nan")
        ),
        "tokenizer_sources": ";".join(sorted(tokenizer_sources)) or "not_recorded",
        "tokenizer_classes": ";".join(sorted(tokenizer_classes)) or "not_recorded",
        "tokenizer_vocab_sizes": ";".join(sorted(tokenizer_vocab_sizes)) or "not_recorded",
    }


def _variant_jsonl_candidates(benchmark_dir: Path, benchmark: str, variant: str) -> List[Path]:
    candidates = [
        benchmark_dir / f"{benchmark}_{variant}.jsonl",
        benchmark_dir / variant / f"{benchmark}_{variant}.jsonl",
    ]
    candidates.extend(sorted(benchmark_dir.glob(f"*{variant}.jsonl")))
    candidates.extend(sorted(benchmark_dir.glob(f"{variant}/*{variant}.jsonl")))
    seen = set()
    out = []
    for path in candidates:
        if path.exists() and path not in seen:
            seen.add(path)
            out.append(path)
    return out


def load_variant_metric(benchmark_dir: Path, benchmark: str, variant: str) -> Optional[Dict[str, object]]:
    jsonl_candidates = _variant_jsonl_candidates(benchmark_dir, benchmark, variant)
    jsonl_path = jsonl_candidates[0] if jsonl_candidates else None

    metric = None
    metric_rows = None
    metric_source = None
    if benchmark == "worldvaluebench":
        emd_candidates = [
            benchmark_dir / f"{benchmark}_{variant}_emd_summary.json",
            benchmark_dir / variant / f"{benchmark}_{variant}_emd_summary.json",
        ]
        emd_candidates.extend(sorted(benchmark_dir.glob(f"*{variant}*emd_summary.json")))
        emd_candidates.extend(sorted(benchmark_dir.glob(f"{variant}/*{variant}*emd_summary.json")))
        for path in emd_candidates:
            if path.exists():
                loaded = _read_emd_summary_json(path)
                if loaded is not None:
                    metric, metric_rows = loaded
                    metric_source = str(path)
                    break
    else:
        summary_candidates = [
            benchmark_dir / f"{benchmark}_eval_summary.csv",
            benchmark_dir / variant / f"{benchmark}_eval_summary.csv",
        ]
        summary_candidates.extend(sorted(benchmark_dir.glob(f"*summary.csv")))
        summary_candidates.extend(sorted(benchmark_dir.glob(f"{variant}/*summary.csv")))
        seen = set()
        for path in summary_candidates:
            if not path.exists() or path in seen:
                continue
            seen.add(path)
            loaded = _read_summary_csv(path, variant)
            if loaded is not None:
                metric, metric_rows = loaded
                metric_source = str(path)
                break

    jsonl_stats = _read_jsonl(jsonl_path, benchmark) if jsonl_path else {}
    if metric is None and jsonl_stats:
        metric = float(jsonl_stats["metric"])
        metric_rows = int(jsonl_stats["rows"])
        metric_source = str(jsonl_path)
    if metric is None:
        return None

    return {
        "metric": metric,
        "rows": metric_rows or int(jsonl_stats.get("rows", 0) or 0),
        "metric_source": metric_source or "",
        "jsonl_path": str(jsonl_path) if jsonl_path else "",
        "jsonl_rows": jsonl_stats.get("jsonl_rows", ""),
        "score_rows": jsonl_stats.get("score_rows", ""),
        "constant_score_rate": jsonl_stats.get("constant_score_rate", ""),
        "first_option_prediction_rate": jsonl_stats.get("first_option_prediction_rate", ""),
        "tokenizer_sources": jsonl_stats.get("tokenizer_sources", "not_recorded"),
        "tokenizer_classes": jsonl_stats.get("tokenizer_classes", "not_recorded"),
        "tokenizer_vocab_sizes": jsonl_stats.get("tokenizer_vocab_sizes", "not_recorded"),
    }


def iter_run_dirs(root: Path, benchmark: str, family: str, protocols: Optional[set]) -> Iterable[Tuple[str, str, Path]]:
    benchmark_root = root / benchmark
    if not benchmark_root.exists():
        return
    for protocol_dir in sorted(p for p in benchmark_root.iterdir() if p.is_dir()):
        if protocols is not None and protocol_dir.name not in protocols:
            continue
        family_dir = protocol_dir / family
        if not family_dir.exists():
            continue
        seed_dirs = sorted(p for p in family_dir.glob("seed_*") if p.is_dir())
        for seed_dir in seed_dirs:
            benchmark_dir = seed_dir / benchmark
            if benchmark_dir.exists():
                yield protocol_dir.name, seed_dir.name, benchmark_dir


def _fmt(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.10g}"
    return "" if value is None else str(value)


def collect_rows(args: argparse.Namespace) -> List[Dict[str, object]]:
    protocols = None
    if args.protocols:
        protocols = {x.strip() for x in args.protocols.split(",") if x.strip()}
    rows: List[Dict[str, object]] = []
    for benchmark in args.benchmarks:
        lower = benchmark in LOWER_IS_BETTER
        metric_name = "overall_emd" if lower else "accuracy"
        direction = "lower_is_better" if lower else "higher_is_better"
        for family in args.families:
            pair = PAIR_SPECS[family]
            for protocol, seed, benchmark_dir in iter_run_dirs(args.root, benchmark, family, protocols):
                pos = load_variant_metric(benchmark_dir, benchmark, pair.positive)
                neg = load_variant_metric(benchmark_dir, benchmark, pair.negative)
                notes: List[str] = []
                valid = pos is not None and neg is not None
                if not valid:
                    notes.append("missing_positive_or_negative")
                if valid:
                    assert pos is not None and neg is not None
                    pos_rows = int(pos.get("rows") or 0)
                    neg_rows = int(neg.get("rows") or 0)
                    if pos_rows < args.min_rows or neg_rows < args.min_rows:
                        valid = False
                        notes.append("below_min_rows")
                    if pos_rows != neg_rows:
                        valid = False
                        notes.append("row_count_mismatch")
                    for side, stats in (("positive", pos), ("negative", neg)):
                        const = stats.get("constant_score_rate")
                        if isinstance(const, float) and math.isfinite(const):
                            if const > args.max_constant_score_rate:
                                valid = False
                                notes.append(f"{side}_constant_scores")
                if valid:
                    assert pos is not None and neg is not None
                    pos_metric = float(pos["metric"])
                    neg_metric = float(neg["metric"])
                    delta = (neg_metric - pos_metric) if lower else (pos_metric - neg_metric)
                    satisfied = delta > 0.0
                else:
                    pos_metric = float(pos["metric"]) if pos is not None else float("nan")
                    neg_metric = float(neg["metric"]) if neg is not None else float("nan")
                    delta = float("nan")
                    satisfied = False

                row = {
                    "benchmark": benchmark,
                    "protocol": protocol,
                    "family": family,
                    "seed": seed,
                    "comparison": pair.label,
                    "metric": metric_name,
                    "direction": direction,
                    "positive_variant": pair.positive,
                    "negative_variant": pair.negative,
                    "positive_metric": pos_metric,
                    "negative_metric": neg_metric,
                    "delta_positive_better": delta,
                    "requirement_satisfied": int(satisfied),
                    "valid_for_comparison": int(valid),
                    "positive_rows": pos.get("rows", "") if pos else "",
                    "negative_rows": neg.get("rows", "") if neg else "",
                    "positive_constant_score_rate": pos.get("constant_score_rate", "") if pos else "",
                    "negative_constant_score_rate": neg.get("constant_score_rate", "") if neg else "",
                    "positive_first_option_prediction_rate": pos.get("first_option_prediction_rate", "") if pos else "",
                    "negative_first_option_prediction_rate": neg.get("first_option_prediction_rate", "") if neg else "",
                    "positive_tokenizer_sources": pos.get("tokenizer_sources", "") if pos else "",
                    "negative_tokenizer_sources": neg.get("tokenizer_sources", "") if neg else "",
                    "positive_tokenizer_classes": pos.get("tokenizer_classes", "") if pos else "",
                    "negative_tokenizer_classes": neg.get("tokenizer_classes", "") if neg else "",
                    "positive_tokenizer_vocab_sizes": pos.get("tokenizer_vocab_sizes", "") if pos else "",
                    "negative_tokenizer_vocab_sizes": neg.get("tokenizer_vocab_sizes", "") if neg else "",
                    "notes": ";".join(dict.fromkeys(notes)),
                    "positive_jsonl": pos.get("jsonl_path", "") if pos else "",
                    "negative_jsonl": neg.get("jsonl_path", "") if neg else "",
                    "run_dir": str(benchmark_dir),
                }
                rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field)) for field in fields})


def best_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    best: Dict[Tuple[str, str], Dict[str, object]] = {}
    for row in rows:
        if not row.get("valid_for_comparison"):
            continue
        delta = row.get("delta_positive_better")
        if not isinstance(delta, float) or not math.isfinite(delta):
            continue
        key = (str(row["benchmark"]), str(row["family"]))
        if key not in best or delta > float(best[key]["delta_positive_better"]):
            best[key] = row
    return [best[key] for key in sorted(best)]


def print_best(rows: List[Dict[str, object]]) -> None:
    selected = best_rows(rows)
    if not selected:
        print("[!] No valid paired probe comparisons found.")
        return
    print("[best valid paired probes]")
    for row in selected:
        status = "SAT" if row.get("requirement_satisfied") else "NO"
        print(
            f"  {row['benchmark']} {row['family']} {row['protocol']} {row['seed']}: "
            f"{status} delta={float(row['delta_positive_better']):.6f} "
            f"pos={float(row['positive_metric']):.6f} neg={float(row['negative_metric']):.6f} "
            f"rows={row['positive_rows']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl external protocol probes and compare paired variants.")
    parser.add_argument("--root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS)
    parser.add_argument("--families", nargs="+", default=["maple_1b", "maple_3b", "llama32_1b"])
    parser.add_argument("--protocols", default=None, help="Optional comma-separated protocol allowlist.")
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--max-constant-score-rate", type=float, default=0.05)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--best-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for family in args.families:
        if family not in PAIR_SPECS:
            raise ValueError(f"Unknown family: {family}")

    timestamp = subprocess.check_output(["date", "+%Y%m%d_%H%M%S"], text=True).strip()
    report_root = args.root / "_reports"
    output_csv = args.output_csv or report_root / f"protocol_probe_matrix_{timestamp}.csv"
    best_csv = args.best_csv or report_root / f"protocol_probe_best_{timestamp}.csv"

    rows = collect_rows(args)
    write_csv(output_csv, rows)
    write_csv(best_csv, best_rows(rows))
    print(f"[wrote] {output_csv}")
    print(f"[wrote] {best_csv}")
    print_best(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
