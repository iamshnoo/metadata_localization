#!/usr/bin/env python3

import csv
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_semantic_gold_ambiguous_1700"
AUDIT_DIR = BASE / "explicit_max_audit"
STRICT_DIR = AUDIT_DIR / "strict_1000"
INPUT_JSONL = STRICT_DIR / "localnewsqa_targetqa_explicit_strict_1000_per_country_polished.jsonl"
INPUT_CSV = STRICT_DIR / "localnewsqa_targetqa_explicit_strict_1000_per_country_polished.csv"
AMBIGUOUS_JSONL = BASE / "localnewsqa_ambiguous_semantic_gold_1700.jsonl"
CANDIDATE_POOLS = [
    AUDIT_DIR / "localnewsqa_targetqa_explicit_style_max_strict_no_warnings.jsonl",
    AUDIT_DIR / "localnewsqa_targetqa_explicit_style_max_paper_clean.jsonl",
    AUDIT_DIR / "localnewsqa_targetqa_explicit_style_max_clean.jsonl",
]
FETCH_CACHE = AUDIT_DIR / "explicit_target_evidence_fetches.jsonl"
POLISHED_FETCH_CACHE = STRICT_DIR / "polished_audit/explicit_target_evidence_fetches.jsonl"
LOG_PATH = STRICT_DIR / "ambiguous_hardening_explicit_overlap_repair_log.csv"
SUMMARY_PATH = STRICT_DIR / "ambiguous_hardening_explicit_overlap_repair_summary.json"

AUDIT_EXPLICIT_SCRIPT = ROOT / "qa_data/localnewsqa_core/48_audit_explicit_max_split.py"
AUDIT_FETCH_SCRIPT = ROOT / "qa_data/localnewsqa_core/32_web_audit_ambiguous_verifiable.py"

COUNTRY_ORDER = [
    "Bangladesh",
    "Canada",
    "Ghana",
    "Hong Kong",
    "India",
    "Ireland",
    "Jamaica",
    "Kenya",
    "Malaysia",
    "Nigeria",
    "Pakistan",
    "Philippines",
    "South Africa",
    "Sri Lanka",
    "Tanzania",
    "United Kingdom",
    "United States",
]

SOURCE_PRIORITY = {
    "explicit": 0,
    "ambiguous_salvaged_target": 1,
    "ambiguous_pool_salvaged_target_web_supported": 2,
}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
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


def norm(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def candidate_rank(row: dict) -> tuple:
    return (
        SOURCE_PRIORITY.get(row.get("source_split_type", ""), 9),
        norm(row.get("target_answer", "")),
        norm(row.get("question", "")),
        row.get("source_row_id", ""),
    )


def source_composition(rows: list[dict]) -> dict[str, int]:
    return dict(Counter(row.get("source_split_type", "") for row in rows))


def main() -> None:
    explicit_audit = load_module(AUDIT_EXPLICIT_SCRIPT, "audit48")
    fetch_audit = load_module(AUDIT_FETCH_SCRIPT, "audit32")
    fetches = {}
    fetches.update(explicit_audit.load_fetch_cache(FETCH_CACHE))
    fetches.update(explicit_audit.load_fetch_cache(POLISHED_FETCH_CACHE))

    ambiguous = read_jsonl(AMBIGUOUS_JSONL)
    ambiguous_ids = {row["source_row_id"] for row in ambiguous}
    rows = read_jsonl(INPUT_JSONL)
    original_rows = len(rows)
    overlaps = [row for row in rows if row.get("source_row_id") in ambiguous_ids]
    kept = [row for row in rows if row.get("source_row_id") not in ambiguous_ids]

    used_sources = {row["source_row_id"] for row in kept}
    used_questions = {norm(row.get("question", "")) for row in kept}
    candidates_by_country: dict[str, list[dict]] = defaultdict(list)
    seen_candidate_sources = set()
    for pool_path in CANDIDATE_POOLS:
        for candidate in read_jsonl(pool_path):
            source_id = candidate.get("source_row_id", "")
            if source_id in seen_candidate_sources:
                continue
            seen_candidate_sources.add(source_id)
            if source_id in ambiguous_ids or source_id in used_sources:
                continue
            if norm(candidate.get("question", "")) in used_questions:
                continue
            country = candidate.get("country", "")
            if country in COUNTRY_ORDER:
                candidates_by_country[country].append(candidate)
    for country in candidates_by_country:
        candidates_by_country[country].sort(key=candidate_rank)

    replacements = []
    rejected_candidates = []
    final_rows = list(kept)
    counts = Counter(row["country"] for row in final_rows)
    for country in COUNTRY_ORDER:
        deficit = 1000 - counts[country]
        if deficit <= 0:
            continue
        for candidate in candidates_by_country[country]:
            if deficit <= 0:
                break
            source_id = candidate.get("source_row_id", "")
            question_key = norm(candidate.get("question", ""))
            if source_id in used_sources or question_key in used_questions:
                continue
            failures, warnings = explicit_audit.audit_explicit_row(candidate, fetches, ambiguous_ids, fetch_audit)
            if failures or warnings:
                rejected_candidates.append(
                    {
                        "country": country,
                        "source_row_id": source_id,
                        "target_answer": candidate.get("target_answer", ""),
                        "target_evidence_title": candidate.get("target_evidence_title", ""),
                        "failures": " | ".join(failures),
                        "warnings": " | ".join(warnings),
                    }
                )
                continue
            final_rows.append(candidate)
            used_sources.add(source_id)
            used_questions.add(question_key)
            replacements.append(
                {
                    "country": country,
                    "source_row_id": source_id,
                    "question": candidate.get("question", ""),
                    "target_answer": candidate.get("target_answer", ""),
                    "source_split_type": candidate.get("source_split_type", ""),
                    "target_evidence_title": candidate.get("target_evidence_title", ""),
                    "target_evidence_url": candidate.get("target_evidence_url", ""),
                }
            )
            counts[country] += 1
            deficit -= 1

    final_rows.sort(key=lambda row: (COUNTRY_ORDER.index(row["country"]), candidate_rank(row)))
    final_sources = [row.get("source_row_id", "") for row in final_rows]
    final_questions = [norm(row.get("question", "")) for row in final_rows]
    validation_errors = []
    country_counts = Counter(row["country"] for row in final_rows)
    for country in COUNTRY_ORDER:
        if country_counts[country] != 1000:
            validation_errors.append(f"{country}: expected 1000, got {country_counts[country]}")
    if len(final_rows) != 17000:
        validation_errors.append(f"expected 17000 rows, got {len(final_rows)}")
    overlap_after = sorted(set(final_sources) & ambiguous_ids)
    if overlap_after:
        validation_errors.append(f"ambiguous source overlap remains: {len(overlap_after)}")
    if len(final_sources) != len(set(final_sources)):
        validation_errors.append("duplicate source ids remain")
    if len(final_questions) != len(set(final_questions)):
        validation_errors.append("duplicate questions remain")

    write_jsonl(INPUT_JSONL, final_rows)
    write_csv(INPUT_CSV, final_rows)
    write_csv(LOG_PATH, replacements)

    summary = {
        "original_rows": original_rows,
        "removed_overlap_rows": len(overlaps),
        "removed_overlap_by_country": dict(Counter(row["country"] for row in overlaps)),
        "replacement_rows": len(replacements),
        "replacement_by_country": dict(Counter(row["country"] for row in replacements)),
        "rejected_candidate_checks": len(rejected_candidates),
        "rows": len(final_rows),
        "country_counts": dict(country_counts),
        "source_composition": source_composition(final_rows),
        "ambiguous_source_overlap_after": len(overlap_after),
        "duplicate_source_ids": len(final_sources) - len(set(final_sources)),
        "duplicate_questions": len(final_questions) - len(set(final_questions)),
        "valid": not validation_errors,
        "validation_errors": validation_errors,
        "paths": {
            "jsonl": str(INPUT_JSONL),
            "csv": str(INPUT_CSV),
            "repair_log": str(LOG_PATH),
            "summary": str(SUMMARY_PATH),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if validation_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
