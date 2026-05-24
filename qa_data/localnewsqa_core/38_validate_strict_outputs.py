#!/usr/bin/env python3

import argparse
import csv
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_strict_pass_ambiguous_1700"
)
DEFAULT_AMBIGUOUS = DEFAULT_DIR / "localnewsqa_ambiguous_web_repaired_1700.jsonl"
DEFAULT_EXPLICIT = DEFAULT_DIR / "localnewsqa_targetqa_explicit_style_max_target_supported.jsonl"
DEFAULT_CACHE = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_audit_ambiguous_1700/url_fetches.jsonl"
)
AUDIT_SCRIPT = ROOT / "qa_data/localnewsqa_core/32_web_audit_ambiguous_verifiable.py"


def load_audit_module():
    spec = importlib.util.spec_from_file_location("audit32", AUDIT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Validate strict ambiguous core and max explicit split.")
    parser.add_argument("--ambiguous", type=Path, default=DEFAULT_AMBIGUOUS)
    parser.add_argument("--explicit", type=Path, default=DEFAULT_EXPLICIT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()

    audit_mod = load_audit_module()
    fetches = audit_mod.load_existing_fetches(args.cache)
    ambiguous_rows = read_jsonl(args.ambiguous)
    explicit_rows = read_jsonl(args.explicit)
    audits = [audit_mod.audit_row(row, fetches) for row in ambiguous_rows]
    failures = [row for row in audits if row["severity"] == "fail"]
    warnings = [row for row in audits if row["severity"] == "warn"]

    errors = []
    country_counts = Counter(row["country"] for row in ambiguous_rows)
    if len(ambiguous_rows) != 1700:
        errors.append(f"strict ambiguous expected 1700 rows, got {len(ambiguous_rows)}")
    for country, count in sorted(country_counts.items()):
        if count != 100:
            errors.append(f"{country}: strict ambiguous expected 100 rows, got {count}")
    if len(country_counts) != 17:
        errors.append(f"strict ambiguous expected 17 countries, got {len(country_counts)}")
    ambiguous_source_ids = [row["source_row_id"] for row in ambiguous_rows]
    explicit_source_ids = [row["source_row_id"] for row in explicit_rows]
    if len(ambiguous_source_ids) != len(set(ambiguous_source_ids)):
        errors.append("duplicate source_row_id in strict ambiguous")
    if len(explicit_source_ids) != len(set(explicit_source_ids)):
        errors.append("duplicate source_row_id in max explicit")
    overlap = set(ambiguous_source_ids) & set(explicit_source_ids)
    if overlap:
        errors.append(f"strict ambiguous/max explicit overlap: {len(overlap)} source ids")
    if failures:
        errors.append(f"strict ambiguous has {len(failures)} web-audit failures")
    if warnings:
        errors.append(f"strict ambiguous has {len(warnings)} web-audit warnings")

    explicit_additions = [
        row
        for row in explicit_rows
        if row.get("source_split_type") == "ambiguous_pool_salvaged_target_web_supported"
    ]
    target_fail_additions = [
        row for row in explicit_additions if row.get("target_web_audit_severity") not in {"pass", "warn"}
    ]
    if target_fail_additions:
        errors.append(f"max explicit has {len(target_fail_additions)} unsupported pool additions")

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    audit_rows_path = outdir / "strict_ambiguous_final_web_audit_rows.csv"
    write_csv(audit_rows_path, audits)
    summary_path = outdir / "final_validation_summary.json"
    summary = {
        "strict_ambiguous_jsonl": str(args.ambiguous),
        "max_explicit_jsonl": str(args.explicit),
        "fetch_cache": str(args.cache),
        "valid": not errors,
        "validation_errors": errors,
        "strict_ambiguous_rows": len(ambiguous_rows),
        "strict_ambiguous_country_counts": dict(country_counts),
        "strict_ambiguous_web_audit_severity_counts": dict(Counter(row["severity"] for row in audits)),
        "strict_ambiguous_failure_counts": dict(
            Counter(mode for row in failures for mode in row["failures"].split(" | ") if mode)
        ),
        "strict_ambiguous_warning_counts": dict(
            Counter(mode for row in warnings for mode in row["warnings"].split(" | ") if mode)
        ),
        "max_explicit_rows": len(explicit_rows),
        "max_explicit_source_split_type_counts": dict(
            Counter(row.get("source_split_type", "") for row in explicit_rows)
        ),
        "max_explicit_pool_addition_rows": len(explicit_additions),
        "max_explicit_pool_addition_target_severity_counts": dict(
            Counter(row.get("target_web_audit_severity", "") for row in explicit_additions)
        ),
        "strict_ambiguous_max_explicit_overlap": len(overlap),
        "sha256": {
            "strict_ambiguous_jsonl": sha256(args.ambiguous),
            "max_explicit_jsonl": sha256(args.explicit),
        },
        "paths": {
            "strict_ambiguous_final_web_audit_rows": str(audit_rows_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    result = validate()
    if not result["valid"]:
        raise SystemExit(1)
