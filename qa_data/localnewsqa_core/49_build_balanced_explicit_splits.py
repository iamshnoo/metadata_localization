#!/usr/bin/env python3

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STRICT_DIR = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_semantic_gold_ambiguous_1700"
)
AUDIT_DIR = STRICT_DIR / "explicit_max_audit"
DEFAULT_INPUT = AUDIT_DIR / "localnewsqa_targetqa_explicit_style_max_paper_clean.jsonl"
DEFAULT_AMBIGUOUS = STRICT_DIR / "localnewsqa_ambiguous_semantic_gold_1700.jsonl"
DEFAULT_FAILURES = AUDIT_DIR / "explicit_max_quality_failures.csv"
DEFAULT_WARNINGS = AUDIT_DIR / "explicit_max_quality_warnings.csv"
DEFAULT_OUTDIR = AUDIT_DIR / "balanced"

CAPS = [1083, 1200, 1300, 1400, 1500]
SOURCE_PRIORITY = {
    "explicit": 0,
    "ambiguous_salvaged_target": 1,
    "ambiguous_pool_salvaged_target_web_supported": 2,
}
REVIEWER_VISIBLE_WARNINGS = {
    "duplicate_question_text",
    "target_country_marker_not_found_in_evidence",
    "target_evidence_text_short",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
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


def norm(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_issue_csv(path: Path, column: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    if not path.exists() or path.stat().st_size == 0:
        return out
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source_id = row.get("source_row_id", "")
            issues = {part.strip() for part in row.get(column, "").split("|") if part.strip()}
            if source_id and issues:
                out[source_id] = issues
    return out


def is_reviewer_visible_warning(warning: str) -> bool:
    return warning in REVIEWER_VISIBLE_WARNINGS or warning.startswith("generic_target_evidence:")


def row_rank(row: dict[str, Any], warnings: dict[str, set[str]]) -> tuple[Any, ...]:
    row_warnings = warnings.get(row.get("source_row_id", ""), set())
    return (
        len(row_warnings),
        SOURCE_PRIORITY.get(row.get("source_split_type", ""), 9),
        norm(row.get("target_answer", "")),
        norm(row.get("question", "")),
        row.get("source_row_id", ""),
    )


def round_robin_by_answer(rows: list[dict[str, Any]], warnings: dict[str, set[str]], limit: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[norm(row.get("target_answer", ""))].append(row)
    active = []
    for answer, bucket in buckets.items():
        bucket.sort(key=lambda row: row_rank(row, warnings))
        active.append([answer, bucket])

    selected: list[dict[str, Any]] = []
    while active and len(selected) < limit:
        active.sort(key=lambda item: (row_rank(item[1][0], warnings), item[0]))
        next_active = []
        for answer, bucket in active:
            selected.append(bucket.pop(0))
            if len(selected) >= limit:
                if bucket:
                    next_active.append([answer, bucket])
                break
            if bucket:
                next_active.append([answer, bucket])
        active = next_active
    return selected


def select_country_rows(rows: list[dict[str, Any]], warnings: dict[str, set[str]], quota: int) -> list[dict[str, Any]]:
    strict = [row for row in rows if not warnings.get(row.get("source_row_id", ""), set())]
    warned = [row for row in rows if warnings.get(row.get("source_row_id", ""), set())]
    selected = round_robin_by_answer(strict, warnings, quota)
    if len(selected) < quota:
        selected.extend(round_robin_by_answer(warned, warnings, quota - len(selected)))
    selected.sort(key=lambda row: (row.get("country", ""), row_rank(row, warnings)))
    return selected


def verification_summary(
    rows: list[dict[str, Any]],
    ambiguous_ids: set[str],
    failures: dict[str, set[str]],
    warnings: dict[str, set[str]],
) -> dict[str, Any]:
    ids = [row.get("source_row_id", "") for row in rows]
    questions = [row.get("question", "") for row in rows]
    country_counts = Counter(row.get("country", "") for row in rows)
    source_counts = Counter(row.get("source_split_type", "") for row in rows)
    hard_failure_ids = sorted(source_id for source_id in ids if failures.get(source_id))
    reviewer_visible_warning_ids = sorted(
        source_id
        for source_id in ids
        if any(is_reviewer_visible_warning(warning) for warning in warnings.get(source_id, set()))
    )
    warning_counts = Counter(
        warning
        for source_id in ids
        for warning in warnings.get(source_id, set())
        if warning
    )
    strict_rows = sum(1 for source_id in ids if not warnings.get(source_id, set()))
    min_count = min(country_counts.values()) if country_counts else 0
    max_count = max(country_counts.values()) if country_counts else 0
    return {
        "rows": len(rows),
        "country_counts": dict(sorted(country_counts.items())),
        "country_min": min_count,
        "country_max": max_count,
        "country_max_min_ratio": round(max_count / min_count, 6) if min_count else None,
        "source_split_type_counts": dict(sorted(source_counts.items())),
        "strict_no_warning_rows": strict_rows,
        "fetch_metadata_warning_rows": len(rows) - strict_rows,
        "warning_counts": dict(sorted(warning_counts.items())),
        "overlap_with_ambiguous_ids": len(set(ids) & ambiguous_ids),
        "duplicate_source_id_count": sum(1 for count in Counter(ids).values() if count > 1),
        "duplicate_question_text_count": sum(1 for count in Counter(questions).values() if count > 1),
        "hard_failure_rows": len(hard_failure_ids),
        "reviewer_visible_warning_rows": len(reviewer_visible_warning_ids),
        "hard_failure_examples": hard_failure_ids[:10],
        "reviewer_visible_warning_examples": reviewer_visible_warning_ids[:10],
    }


def main() -> None:
    rows = read_jsonl(DEFAULT_INPUT)
    ambiguous_ids = {row.get("source_row_id", "") for row in read_jsonl(DEFAULT_AMBIGUOUS)}
    failures = read_issue_csv(DEFAULT_FAILURES, "failures")
    warnings = read_issue_csv(DEFAULT_WARNINGS, "warnings")

    by_country: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_country[row.get("country", "")].append(row)

    min_country_count = min(len(country_rows) for country_rows in by_country.values())
    caps = sorted(set([min_country_count, *CAPS]))
    DEFAULT_OUTDIR.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for cap in caps:
        selected: list[dict[str, Any]] = []
        for country in sorted(by_country):
            country_rows = by_country[country]
            quota = min(len(country_rows), cap)
            selected.extend(select_country_rows(country_rows, warnings, quota))
        selected.sort(key=lambda row: (row.get("country", ""), row_rank(row, warnings)))
        name = "exact_balanced_max" if cap == min_country_count else f"cap_{cap}"
        jsonl_path = DEFAULT_OUTDIR / f"localnewsqa_targetqa_explicit_paper_clean_{name}.jsonl"
        csv_path = DEFAULT_OUTDIR / f"localnewsqa_targetqa_explicit_paper_clean_{name}.csv"
        write_jsonl(jsonl_path, selected)
        write_csv(csv_path, selected)
        summary = verification_summary(selected, ambiguous_ids, failures, warnings)
        summary["cap"] = cap
        summary["jsonl"] = str(jsonl_path)
        summary["csv"] = str(csv_path)
        summary["policy"] = (
            "country-wise deterministic selection from the audited paper-clean explicit pool; "
            "zero-warning rows are selected before fetch/cache-warning rows, and rows are "
            "round-robined by normalized target answer within each country for diversity"
        )
        summaries[name] = summary

    summary_path = DEFAULT_OUTDIR / "balanced_explicit_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")

    rows_for_csv = []
    for name, summary in summaries.items():
        rows_for_csv.append(
            {
                "variant": name,
                "rows": summary["rows"],
                "country_min": summary["country_min"],
                "country_max": summary["country_max"],
                "country_max_min_ratio": summary["country_max_min_ratio"],
                "strict_no_warning_rows": summary["strict_no_warning_rows"],
                "fetch_metadata_warning_rows": summary["fetch_metadata_warning_rows"],
                "overlap_with_ambiguous_ids": summary["overlap_with_ambiguous_ids"],
                "duplicate_source_id_count": summary["duplicate_source_id_count"],
                "duplicate_question_text_count": summary["duplicate_question_text_count"],
                "hard_failure_rows": summary["hard_failure_rows"],
                "reviewer_visible_warning_rows": summary["reviewer_visible_warning_rows"],
                "jsonl": summary["jsonl"],
                "csv": summary["csv"],
            }
        )
    write_csv(DEFAULT_OUTDIR / "balanced_explicit_summary.csv", rows_for_csv)
    print(json.dumps(summaries, indent=2, sort_keys=True))

    bad = [
        name
        for name, summary in summaries.items()
        if summary["overlap_with_ambiguous_ids"]
        or summary["duplicate_source_id_count"]
        or summary["duplicate_question_text_count"]
        or summary["hard_failure_rows"]
        or summary["reviewer_visible_warning_rows"]
    ]
    if bad:
        raise SystemExit(f"balanced variants failed verification: {', '.join(bad)}")


if __name__ == "__main__":
    main()
