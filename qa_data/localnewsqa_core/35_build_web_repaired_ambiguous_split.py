#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CURRENT = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/localnewsqa_ambiguous_verifiable_1700.jsonl"
)
DEFAULT_POOL = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/localnewsqa_ambiguous_verifiable_pool_4625.jsonl"
)
DEFAULT_CACHE = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_audit_ambiguous_1700/url_fetches.jsonl"
)
DEFAULT_OUTDIR = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_repaired_ambiguous_1700"
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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def csv_value(value: Any) -> str:
    if isinstance(value, list):
        return " || ".join(str(part) for part in value)
    return "" if value is None else str(value)


def flatten_for_csv(row: dict) -> dict:
    return {key: csv_value(value) for key, value in row.items()}


def sort_key(item: dict) -> tuple:
    severity_rank = {"pass": 0, "warn": 1}
    return (
        item.get("country", ""),
        severity_rank.get(item.get("web_audit_severity", ""), 9),
        str(item.get("weak_locale_risk", "")) != "clean",
        not bool(item.get("llm_accept_or_restored", False)),
        -float(item.get("evidence_support_score", 0) or 0),
        float(item.get("semantic_error_probability", 1) or 1),
        item.get("source_row_id", ""),
    )


def attach_audit(row: dict, audit: dict, origin: str) -> dict:
    out = dict(row)
    out["web_audit_severity"] = audit["severity"]
    out["web_audit_failures"] = audit["failures"]
    out["web_audit_warnings"] = audit["warnings"]
    out["web_repair_origin"] = origin
    out["target_evidence_final_url"] = audit.get("target_final_url", "")
    out["contrast_evidence_final_url"] = audit.get("contrast_final_url", "")
    out["target_evidence_fetch_source"] = audit.get("target_source", "")
    out["contrast_evidence_fetch_source"] = audit.get("contrast_source", "")
    return out


def validate(rows: list[dict], countries: list[str], allowed_severities: set[str]) -> list[str]:
    errors = []
    counts = Counter(row["country"] for row in rows)
    if len(rows) != 1700:
        errors.append(f"expected 1700 rows, got {len(rows)}")
    for country in countries:
        if counts[country] != 100:
            errors.append(f"{country}: expected 100 rows, got {counts[country]}")
    source_ids = [row["source_row_id"] for row in rows]
    if len(source_ids) != len(set(source_ids)):
        errors.append("duplicate source_row_id values in repaired split")
    hard_failures = [
        row
        for row in rows
        if row.get("web_audit_severity") not in allowed_severities or row.get("web_audit_failures")
    ]
    if hard_failures:
        errors.append(f"{len(hard_failures)} rows do not satisfy allowed severities {sorted(allowed_severities)}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a no-hard-failure web-repaired ambiguous split.")
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--strict-pass-only", action="store_true")
    args = parser.parse_args()

    audit_mod = load_audit_module()
    fetches = audit_mod.load_existing_fetches(args.cache)
    current_rows = read_jsonl(args.current)
    pool_rows = read_jsonl(args.pool)
    countries = sorted({row["country"] for row in current_rows})
    allowed_severities = {"pass"} if args.strict_pass_only else {"pass", "warn"}

    current_selected = []
    dropped_current = []
    current_source_ids = set()
    for row in current_rows:
        audit = audit_mod.audit_row(row, fetches)
        current_source_ids.add(row["source_row_id"])
        if audit["severity"] in allowed_severities:
            current_selected.append(attach_audit(row, audit, "original_1700"))
        else:
            dropped_current.append(attach_audit(row, audit, "dropped_original_1700"))

    selected = list(current_selected)
    selected_source_ids = {row["source_row_id"] for row in selected}
    need = {country: 100 - Counter(row["country"] for row in selected)[country] for country in countries}

    candidates = []
    for row in pool_rows:
        if row["source_row_id"] in current_source_ids:
            continue
        if need.get(row["country"], 0) <= 0:
            continue
        if row.get("target_evidence_url") not in fetches or row.get("contrast_evidence_url") not in fetches:
            continue
        audit = audit_mod.audit_row(row, fetches)
        if audit["severity"] in allowed_severities:
            candidates.append(attach_audit(row, audit, "pool_replacement"))

    candidates.sort(key=sort_key)
    replacements = []
    counts = Counter(row["country"] for row in selected)
    for row in candidates:
        country = row["country"]
        if counts[country] >= 100:
            continue
        if row["source_row_id"] in selected_source_ids:
            continue
        selected.append(row)
        replacements.append(row)
        selected_source_ids.add(row["source_row_id"])
        counts[country] += 1

    selected.sort(key=lambda row: (row["country"], sort_key(row), row["source_row_id"]))
    final_rows = []
    for idx, row in enumerate(selected, start=1):
        out = dict(row)
        out["id"] = f"localnewsqa_ambig_web_verified_{idx:04d}"
        final_rows.append(out)

    errors = validate(final_rows, countries, allowed_severities)
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.outdir / "localnewsqa_ambiguous_web_repaired_1700.jsonl", final_rows)
    write_csv(args.outdir / "localnewsqa_ambiguous_web_repaired_1700.csv", [flatten_for_csv(row) for row in final_rows])
    write_csv(args.outdir / "dropped_original_fail_rows.csv", [flatten_for_csv(row) for row in dropped_current])
    write_csv(args.outdir / "pool_replacement_rows.csv", [flatten_for_csv(row) for row in replacements])

    summary = {
        "current_input": str(args.current),
        "pool_input": str(args.pool),
        "fetch_cache": str(args.cache),
        "rows": len(final_rows),
        "strict_pass_only": args.strict_pass_only,
        "allowed_web_audit_severities": sorted(allowed_severities),
        "validation_errors": errors,
        "valid": not errors,
        "country_counts": dict(Counter(row["country"] for row in final_rows)),
        "origin_counts": dict(Counter(row["web_repair_origin"] for row in final_rows)),
        "web_audit_severity_counts": dict(Counter(row["web_audit_severity"] for row in final_rows)),
        "warning_counts": dict(Counter(mode for row in final_rows for mode in row["web_audit_warnings"].split(" | ") if mode)),
        "dropped_original_fail_rows": len(dropped_current),
        "replacement_rows": len(replacements),
        "replacement_by_country": dict(Counter(row["country"] for row in replacements)),
        "paths": {
            "jsonl": str(args.outdir / "localnewsqa_ambiguous_web_repaired_1700.jsonl"),
            "csv": str(args.outdir / "localnewsqa_ambiguous_web_repaired_1700.csv"),
            "dropped_original_fail_rows": str(args.outdir / "dropped_original_fail_rows.csv"),
            "pool_replacement_rows": str(args.outdir / "pool_replacement_rows.csv"),
            "summary": str(args.outdir / "summary.json"),
        },
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
