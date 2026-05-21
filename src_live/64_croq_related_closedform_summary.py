#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional


BENCHMARKS = ("blend", "globalmmlu", "globalmmlu_cs", "normad")

CUSTOM_LABELS = {
    "custom_tplus_eplus": "(T+, I+)",
    "custom_tplus_eminus": "(T+, I-)",
    "custom_tminus_eplus": "(T-, I+)",
    "custom_tminus_eminus": "(T-, I-)",
}

HF_LABELS = {
    "llama3_chat_with_metadata": "I+",
    "llama3_chat_without_metadata": "I-",
}

MODEL_NAME_MAP = {
    "llama32_1b": "LLaMA-3.2-1B",
    "llama32_3b": "LLaMA-3.2-3B",
    "qwen25_0p5b": "Qwen2.5-0.5B",
    "qwen25_1p5b": "Qwen2.5-1.5B",
    "qwen25_3b": "Qwen2.5-3B",
    "qwen35_0p8b": "Qwen3.5-0.8B",
    "qwen35_2b": "Qwen3.5-2B",
    "qwen35_4b": "Qwen3.5-4B",
    "qwen35_9b": "Qwen3.5-9B",
    "gemma4_e2b_it": "Gemma-4-E2B-it",
    "gemma4_e4b_it": "Gemma-4-E4B-it",
    "mistral7b_v02": "Mistral-7B-Instruct-v0.2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate CROQ-related closed-form benchmark runs."
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        required=True,
        help="Run roots to scan.",
    )
    parser.add_argument(
        "--output-long-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-wide-csv",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_summary(path: Path) -> Optional[Dict[str, str]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def detect_record(path: Path) -> Dict[str, str]:
    parts = path.parts
    benchmark = next(b for b in BENCHMARKS if f"/{b}/" in str(path))
    variant = path.parent.name
    seed_tag = next((p for p in parts if p.startswith("seed_")), "")
    root_label = ""
    model_group = ""
    series = ""
    family = ""

    path_str = str(path)
    if "external_benchmarks_croq_related_hf_seed41_fixed" in path_str:
        root_label = "hf_seed41_fixed"
        anchor = "external_benchmarks_croq_related_hf_seed41_fixed"
        model_group = path.parts[path.parts.index(anchor) + 1]
        series = HF_LABELS.get(variant, variant)
        family = MODEL_NAME_MAP.get(model_group, model_group)
    elif "external_benchmarks_croq_related_hf_seed41_globalmmlu" in path_str:
        root_label = "hf_seed41_globalmmlu"
        anchor = "external_benchmarks_croq_related_hf_seed41_globalmmlu"
        model_group = path.parts[path.parts.index(anchor) + 1]
        series = HF_LABELS.get(variant, variant)
        family = MODEL_NAME_MAP.get(model_group, model_group)
    elif "external_benchmarks_croq_related_hf_seed41" in path_str:
        root_label = "hf_seed41"
        anchor = "external_benchmarks_croq_related_hf_seed41"
        model_group = path.parts[path.parts.index(anchor) + 1]
        series = HF_LABELS.get(variant, variant)
        family = MODEL_NAME_MAP.get(model_group, model_group)
    elif "external_benchmarks_croq_related_maple_protocol_search_seed41_v2" in path_str:
        anchor = "external_benchmarks_croq_related_maple_protocol_search_seed41_v2"
        idx = path.parts.index(anchor)
        model_group = path.parts[idx + 1]
        config_slug = path.parts[idx + 2]
        root_label = f"maple_search_v2_{config_slug}"
        series = CUSTOM_LABELS.get(variant, variant)
        family = "MAPLE 1B" if model_group == "raw_1b" else "MAPLE 3B"
    elif "external_benchmarks_croq_related_maple_gpuq_probe_seed41" in path_str:
        anchor = "external_benchmarks_croq_related_maple_gpuq_probe_seed41"
        idx = path.parts.index(anchor)
        config_slug = path.parts[idx + 1]
        root_label = f"maple_probe_{config_slug}"
        model_group = "raw_3b"
        series = CUSTOM_LABELS.get(variant, variant)
        family = "MAPLE 3B"
    elif "external_benchmarks_croq_related_maple_protocol_search_seed41" in path_str:
        anchor = "external_benchmarks_croq_related_maple_protocol_search_seed41"
        idx = path.parts.index(anchor)
        model_group = path.parts[idx + 1]
        config_slug = path.parts[idx + 2]
        root_label = f"maple_search_{config_slug}"
        series = CUSTOM_LABELS.get(variant, variant)
        family = "MAPLE 1B" if model_group == "raw_1b" else "MAPLE 3B"
    elif "external_benchmarks_croq_related_seed41_globalmmlu" in path_str:
        root_label = "maple_seed41_globalmmlu"
        model_group = "raw_1b" if "/raw_1b/" in path_str else "raw_3b"
        series = CUSTOM_LABELS.get(variant, variant)
        family = "MAPLE 1B" if model_group == "raw_1b" else "MAPLE 3B"
    elif "external_benchmarks_croq_related_seed41_rerun" in path_str:
        root_label = "maple_seed41_rerun"
        model_group = "raw_1b" if "/raw_1b/" in path_str else "raw_3b"
        series = CUSTOM_LABELS.get(variant, variant)
        family = "MAPLE 1B" if model_group == "raw_1b" else "MAPLE 3B"
    elif "external_benchmarks_croq_related_seed41" in path_str:
        root_label = "maple_seed41"
        model_group = "raw_1b" if "/raw_1b/" in path_str else "raw_3b"
        series = CUSTOM_LABELS.get(variant, variant)
        family = "MAPLE 1B" if model_group == "raw_1b" else "MAPLE 3B"
    else:
        root_label = path.parts[-7] if len(path.parts) >= 7 else "unknown"
        model_group = path.parts[-6] if len(path.parts) >= 6 else "unknown"
        series = variant
        family = model_group

    return {
        "benchmark": benchmark,
        "variant": variant,
        "seed_tag": seed_tag,
        "root_label": root_label,
        "model_group": model_group,
        "family": family,
        "series": series,
    }


def collect_rows(roots: Iterable[Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for root in roots:
        for path in root.rglob("*_eval_summary.csv"):
            summary = read_summary(path)
            if not summary:
                continue
            meta = detect_record(path)
            rows.append(
                {
                    **meta,
                    "path": str(path),
                    "status": summary.get("status", ""),
                    "correct": int(summary.get("correct", 0) or 0),
                    "processed_total": int(summary.get("processed_total", 0) or 0),
                    "accuracy": float(summary.get("accuracy", 0.0) or 0.0),
                    "skipped": int(summary.get("skipped", 0) or 0),
                    "total_rows": int(summary.get("total_rows", 0) or 0),
                    "error": summary.get("error", ""),
                }
            )
    rows.sort(key=lambda r: (r["benchmark"], r["family"], r["series"]))
    return rows


def write_long(rows: List[Dict[str, object]], path: Path) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "benchmark",
                "family",
                "series",
                "variant",
                "root_label",
                "model_group",
                "seed_tag",
                "accuracy",
                "correct",
                "processed_total",
                "status",
                "skipped",
                "total_rows",
                "error",
                "path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_wide(rows: List[Dict[str, object]], path: Path) -> None:
    ensure_parent(path)
    by_key: Dict[tuple, Dict[str, object]] = {}
    for row in rows:
        if row.get("status") != "ok" or int(row.get("processed_total", 0) or 0) <= 0:
            continue
        key = (row["benchmark"], row["family"], row["root_label"])
        rec = by_key.setdefault(
            key,
            {
                "benchmark": row["benchmark"],
                "family": row["family"],
                "root_label": row["root_label"],
            },
        )
        rec[f"{row['series']}_accuracy"] = row["accuracy"]
        rec[f"{row['series']}_correct"] = row["correct"]
        rec[f"{row['series']}_total"] = row["processed_total"]
    wide_rows = sorted(by_key.values(), key=lambda r: (r["benchmark"], r["family"], r["root_label"]))
    fieldnames = sorted({k for row in wide_rows for k in row.keys()})
    ordered = ["benchmark", "family", "root_label"] + [f for f in fieldnames if f not in {"benchmark", "family", "root_label"}]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(wide_rows)


def main() -> int:
    args = parse_args()
    rows = collect_rows(args.roots)
    write_long(rows, args.output_long_csv)
    write_wide(rows, args.output_wide_csv)
    print(f"[✔] Wrote {args.output_long_csv}")
    print(f"[✔] Wrote {args.output_wide_csv}")
    print(f"[✔] Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
