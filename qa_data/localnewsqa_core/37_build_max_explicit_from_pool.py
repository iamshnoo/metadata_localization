#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STRICT_DIR = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_semantic_gold_ambiguous_1700"
)
DEFAULT_TARGETQA = DEFAULT_STRICT_DIR / "localnewsqa_targetqa_explicit_style_disjoint_from_web_ambiguous.jsonl"
DEFAULT_AMBIGUOUS = DEFAULT_STRICT_DIR / "localnewsqa_ambiguous_semantic_gold_1700.jsonl"
DEFAULT_POOL = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/localnewsqa_ambiguous_verifiable_pool_4625.jsonl"
)
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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def csv_value(value: Any) -> str:
    if isinstance(value, list):
        return " || ".join(str(part) for part in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return "" if value is None else str(value)


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
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def normalize_text(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_count(options: list[str], answer: str) -> int:
    norm = normalize_text(answer)
    return sum(1 for opt in options if normalize_text(opt) == norm)


def question_with_country(country: str, question: str) -> str:
    question = str(question or "").strip()
    display_country = {
        "United States": "the United States",
        "United Kingdom": "the United Kingdom",
        "Philippines": "the Philippines",
    }.get(country, country)
    if not question:
        return f"In {display_country},"
    if country.lower() in question.lower():
        return question
    first = question[0].lower() + question[1:] if len(question) > 1 else question.lower()
    return f"In {display_country}, {first}"


def localize_explicit_row(row: dict) -> dict:
    out = dict(row)
    localized = question_with_country(out.get("country", ""), out.get("question", ""))
    out["question"] = localized
    out["localized_question"] = localized
    return out


def target_side_status(row: dict, audit: dict) -> tuple[str, list[str]]:
    reasons = []
    options = row.get("options") or []
    if len(options) != 4:
        reasons.append("option_count_not_4")
    if answer_count(options, row.get("target_answer", "")) != 1:
        reasons.append("target_answer_not_exactly_once")
    if not audit.get("target_url_ok"):
        reasons.append("target_url_fetch_failed")
    if audit.get("target_url_ok") and not audit.get("target_answer_found"):
        reasons.append("target_answer_not_found_in_evidence")
    if reasons:
        return "fail", reasons
    if not audit.get("target_country_marker_found"):
        return "warn", ["target_country_marker_not_found_in_evidence"]
    return "pass", []


def make_explicit_row(row: dict, audit: dict, status: str, reasons: list[str], ordinal: int) -> dict:
    out = dict(row)
    source_id = row["source_row_id"]
    localized = question_with_country(row.get("country", ""), row.get("question", ""))
    out.update(
        {
            "id": source_id,
            "source_row_id": source_id,
            "question": localized,
            "localized_question": localized,
            "split_name": "LocalNewsQA-TargetQA-Max",
            "split_type": "explicit",
            "split_family": "LocalNewsQA-Core",
            "ambiguity_flag": False,
            "source_split_type": "ambiguous_pool_salvaged_target_web_supported",
            "max_explicit_addition_ordinal": ordinal,
            "target_web_audit_severity": status,
            "target_web_audit_notes": " | ".join(reasons),
            "target_url_ok": audit.get("target_url_ok", ""),
            "target_answer_found": audit.get("target_answer_found", ""),
            "target_country_marker_found": audit.get("target_country_marker_found", ""),
            "target_status_code": audit.get("target_status_code", ""),
            "target_domain": audit.get("target_domain", ""),
            "target_text_len": audit.get("target_text_len", ""),
            "target_fetch_error": audit.get("target_fetch_error", ""),
            "target_page_title": audit.get("target_page_title", ""),
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Max out explicit TargetQA using target-supported leftover pool rows.")
    parser.add_argument("--targetqa", type=Path, default=DEFAULT_TARGETQA)
    parser.add_argument("--ambiguous", type=Path, default=DEFAULT_AMBIGUOUS)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_STRICT_DIR)
    args = parser.parse_args()

    audit_mod = load_audit_module()
    fetches = audit_mod.load_existing_fetches(args.cache)
    target_rows = [localize_explicit_row(row) for row in read_jsonl(args.targetqa)]
    ambiguous_rows = read_jsonl(args.ambiguous)
    pool_rows = read_jsonl(args.pool)

    used_source_ids = {row["source_row_id"] for row in target_rows} | {
        row["source_row_id"] for row in ambiguous_rows
    }
    candidates = [row for row in pool_rows if row["source_row_id"] not in used_source_ids]

    additions = []
    excluded = []
    for row in sorted(candidates, key=lambda r: (r.get("country", ""), r.get("source_row_id", ""))):
        audit = audit_mod.audit_row(row, fetches)
        status, reasons = target_side_status(row, audit)
        if status in {"pass", "warn"}:
            additions.append(make_explicit_row(row, audit, status, reasons, len(additions) + 1))
        else:
            rejected = make_explicit_row(row, audit, status, reasons, 0)
            excluded.append(rejected)

    max_rows = list(target_rows) + additions
    source_ids = [row["source_row_id"] for row in max_rows]
    ambiguous_source_ids = {row["source_row_id"] for row in ambiguous_rows}
    validation_errors = []
    if len(source_ids) != len(set(source_ids)):
        validation_errors.append("duplicate source_row_id in max explicit split")
    overlap = set(source_ids) & ambiguous_source_ids
    if overlap:
        validation_errors.append(f"max explicit overlaps strict ambiguous: {len(overlap)} source ids")
    bad_additions = [row for row in additions if row.get("target_web_audit_severity") not in {"pass", "warn"}]
    if bad_additions:
        validation_errors.append(f"{len(bad_additions)} additions do not have target-side support")

    args.outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.outdir / "localnewsqa_targetqa_explicit_style_max_target_supported.jsonl"
    csv_path = args.outdir / "localnewsqa_targetqa_explicit_style_max_target_supported.csv"
    additions_path = args.outdir / "max_explicit_pool_additions.csv"
    excluded_path = args.outdir / "max_explicit_pool_excluded_target_fail.csv"
    summary_path = args.outdir / "max_explicit_summary.json"
    write_jsonl(jsonl_path, max_rows)
    write_csv(csv_path, max_rows)
    write_csv(additions_path, additions)
    write_csv(excluded_path, excluded)

    summary = {
        "targetqa_input": str(args.targetqa),
        "ambiguous_input": str(args.ambiguous),
        "pool_input": str(args.pool),
        "fetch_cache": str(args.cache),
        "strict_ambiguous_rows": len(ambiguous_rows),
        "base_explicit_rows": len(target_rows),
        "pool_rows": len(pool_rows),
        "leftover_pool_candidates": len(candidates),
        "added_target_supported_rows": len(additions),
        "excluded_target_fail_rows": len(excluded),
        "max_explicit_rows": len(max_rows),
        "validation_errors": validation_errors,
        "valid": not validation_errors,
        "base_source_split_type_counts": dict(Counter(row.get("source_split_type", "") for row in target_rows)),
        "addition_target_web_audit_severity_counts": dict(
            Counter(row.get("target_web_audit_severity", "") for row in additions)
        ),
        "addition_by_country": dict(Counter(row.get("country", "") for row in additions)),
        "excluded_by_country": dict(Counter(row.get("country", "") for row in excluded)),
        "max_source_split_type_counts": dict(Counter(row.get("source_split_type", "") for row in max_rows)),
        "paths": {
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
            "additions": str(additions_path),
            "excluded_target_fail": str(excluded_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if validation_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
