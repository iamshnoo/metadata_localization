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
STRICT1000 = AUDIT_DIR / "strict_1000/localnewsqa_targetqa_explicit_strict_1000_per_country_polished.jsonl"
AMBIGUOUS = BASE / "localnewsqa_ambiguous_semantic_gold_1700.jsonl"
OUTDIR = AUDIT_DIR / "strict_defensible_954"
OUT_JSONL = OUTDIR / "localnewsqa_targetqa_explicit_strict_defensible_954_per_country.jsonl"
OUT_CSV = OUTDIR / "localnewsqa_targetqa_explicit_strict_defensible_954_per_country.csv"
SUMMARY = OUTDIR / "strict_defensible_build_summary.json"
REPAIR_LOG = OUTDIR / "evidence_excerpt_anchor_repair_log.csv"

CANDIDATE_POOLS = [
    STRICT1000,
    AUDIT_DIR / "localnewsqa_targetqa_explicit_style_max_strict_no_warnings.jsonl",
    AUDIT_DIR / "localnewsqa_targetqa_explicit_style_max_paper_clean.jsonl",
    AUDIT_DIR / "localnewsqa_targetqa_explicit_style_max_clean.jsonl",
]
FETCH_CACHES = [
    BASE / "semantic_gold_selected_evidence_fetches.jsonl",
    AUDIT_DIR / "explicit_target_evidence_fetches.jsonl",
    AUDIT_DIR / "strict_1000/polished_audit/explicit_target_evidence_fetches.jsonl",
]

AUDIT_EXPLICIT_SCRIPT = ROOT / "qa_data/localnewsqa_core/48_audit_explicit_max_split.py"
FETCH_AUDIT_SCRIPT = ROOT / "qa_data/localnewsqa_core/32_web_audit_ambiguous_verifiable.py"
BUILDER_SCRIPT = ROOT / "qa_data/localnewsqa_core/45_build_relation_strict_gold_ambiguous.py"

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
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def norm(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def rank(row: dict) -> tuple:
    return (
        SOURCE_PRIORITY.get(row.get("source_split_type", ""), 9),
        norm(row.get("target_answer", "")),
        norm(row.get("question", "")),
        row.get("source_row_id", ""),
    )


def answer_anchor_text(row: dict, fetch: dict) -> str:
    return " ".join(
        str(part or "")
        for part in [
            row.get("target_evidence_title", ""),
            row.get("target_evidence_excerpt", ""),
            fetch.get("title", ""),
            str(fetch.get("text", ""))[:1500],
        ]
    )


def full_evidence_blob(row: dict, fetch: dict) -> str:
    return " ".join(
        str(part or "")
        for part in [
            row.get("target_evidence_url", ""),
            row.get("target_evidence_title", ""),
            row.get("target_evidence_excerpt", ""),
            fetch.get("title", ""),
            fetch.get("text", ""),
        ]
    )


def find_answer_snippet(row: dict, fetch: dict, fetch_audit: Any, radius: int = 360) -> str:
    text = re.sub(r"\s+", " ", str(fetch.get("text", "") or "")).strip()
    if not text:
        return ""
    lower = text.lower()
    aliases = sorted(fetch_audit.answer_aliases(row.get("target_answer", "")), key=len, reverse=True)
    for alias in aliases:
        alias = str(alias or "").strip()
        if not alias:
            continue
        idx = lower.find(alias.lower())
        if idx < 0:
            continue
        start = max(0, idx - radius)
        end = min(len(text), idx + len(alias) + radius)
        snippet = text[start:end].strip()
        snippet = re.sub(r"^[^A-Z0-9]*", "", snippet)
        snippet = re.sub(r"\s+", " ", snippet)
        return snippet[:900]
    return ""


def main() -> None:
    explicit_audit = load_module(AUDIT_EXPLICIT_SCRIPT, "audit48")
    fetch_audit = load_module(FETCH_AUDIT_SCRIPT, "fetch32")
    builder = load_module(BUILDER_SCRIPT, "builder45")
    fetches: dict[str, dict] = {}
    for cache in FETCH_CACHES:
        fetches.update(explicit_audit.load_fetch_cache(cache))

    ambiguous_ids = {row["source_row_id"] for row in read_jsonl(AMBIGUOUS)}
    seen_sources = set()
    candidates_by_country: dict[str, list[dict]] = defaultdict(list)
    rejected = Counter()
    high_conf_cache: dict[tuple, bool] = {}

    def high_conf_issue(row: dict) -> bool:
        key = (
            row.get("question", ""),
            row.get("target_evidence_url", ""),
            row.get("target_evidence_title", ""),
            row.get("target_evidence_excerpt", ""),
        )
        if key in high_conf_cache:
            return high_conf_cache[key]
        text = builder.evidence_cue_text(row, "target", fetches)
        issue = any(cue["norm"] not in text for cue in builder.high_confidence_question_cues(row))
        high_conf_cache[key] = issue
        return issue

    for pool in CANDIDATE_POOLS:
        for row in read_jsonl(pool):
            source_id = row.get("source_row_id", "")
            if source_id in seen_sources:
                continue
            seen_sources.add(source_id)
            if source_id in ambiguous_ids:
                rejected["ambiguous_overlap"] += 1
                continue
            if high_conf_issue(row):
                rejected["high_confidence_question_cue"] += 1
                continue
            failures, warnings = explicit_audit.audit_explicit_row(row, fetches, ambiguous_ids, fetch_audit)
            if failures:
                rejected["audit_failure"] += 1
                continue
            if warnings:
                rejected["audit_warning"] += 1
                continue
            if row.get("country") in COUNTRY_ORDER:
                candidates_by_country[row["country"]].append(dict(row))

    for country in candidates_by_country:
        candidates_by_country[country].sort(key=rank)
    target_per_country = min(len(candidates_by_country[country]) for country in COUNTRY_ORDER)

    selected = []
    used_questions = set()
    used_sources = set()
    for country in COUNTRY_ORDER:
        count = 0
        for row in candidates_by_country[country]:
            if count >= target_per_country:
                break
            source_id = row.get("source_row_id", "")
            question_key = norm(row.get("question", ""))
            if source_id in used_sources or question_key in used_questions:
                continue
            selected.append(row)
            used_sources.add(source_id)
            used_questions.add(question_key)
            count += 1

    repair_log = []
    for row in selected:
        fetch = fetches.get(row.get("target_evidence_url", ""), {})
        anchor_norm = fetch_audit.normalize_text(answer_anchor_text(row, fetch))
        if fetch_audit.contains_any(anchor_norm, fetch_audit.answer_aliases(row.get("target_answer", ""))):
            continue
        snippet = find_answer_snippet(row, fetch, fetch_audit)
        if snippet:
            old_excerpt = row.get("target_evidence_excerpt", "")
            row["target_evidence_excerpt"] = snippet
            repair_log.append(
                {
                    "source_row_id": row.get("source_row_id", ""),
                    "country": row.get("country", ""),
                    "target_answer": row.get("target_answer", ""),
                    "target_evidence_title": row.get("target_evidence_title", ""),
                    "target_evidence_url": row.get("target_evidence_url", ""),
                    "old_excerpt": old_excerpt,
                    "new_excerpt": snippet,
                }
            )

    selected.sort(key=lambda row: (COUNTRY_ORDER.index(row["country"]), rank(row)))
    for idx, row in enumerate(selected, start=1):
        row["id"] = f"localnewsqa_explicit_strict_defensible_{idx:05d}"
        row["split_name"] = "LocalNewsQA-Explicit-Strict-Defensible-954"

    country_counts = Counter(row["country"] for row in selected)
    source_ids = [row.get("source_row_id", "") for row in selected]
    question_keys = [norm(row.get("question", "")) for row in selected]
    validation_errors = []
    if len(set(country_counts.values())) != 1:
        validation_errors.append(f"not exactly balanced: {dict(country_counts)}")
    if len(source_ids) != len(set(source_ids)):
        validation_errors.append("duplicate source ids")
    if len(question_keys) != len(set(question_keys)):
        validation_errors.append("duplicate questions")
    if set(source_ids) & ambiguous_ids:
        validation_errors.append(f"ambiguous overlap: {len(set(source_ids) & ambiguous_ids)}")

    write_jsonl(OUT_JSONL, selected)
    write_csv(OUT_CSV, selected)
    write_csv(REPAIR_LOG, repair_log)
    summary = {
        "rows": len(selected),
        "target_per_country": target_per_country,
        "country_counts": dict(country_counts),
        "candidate_counts": {country: len(candidates_by_country[country]) for country in COUNTRY_ORDER},
        "source_split_type_counts": dict(Counter(row.get("source_split_type", "") for row in selected)),
        "rejected_counts": dict(rejected),
        "evidence_excerpt_anchor_repairs": len(repair_log),
        "duplicate_source_ids": len(source_ids) - len(set(source_ids)),
        "duplicate_questions": len(question_keys) - len(set(question_keys)),
        "ambiguous_overlap": len(set(source_ids) & ambiguous_ids),
        "valid": not validation_errors,
        "validation_errors": validation_errors,
        "paths": {
            "jsonl": str(OUT_JSONL),
            "csv": str(OUT_CSV),
            "repair_log": str(REPAIR_LOG),
            "summary": str(SUMMARY),
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if validation_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
