#!/usr/bin/env python3

import csv
import importlib.util
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_semantic_gold_ambiguous_1700"
FINAL_DIR = BASE / "explicit_max_audit/strict_defensible_1000_curated_final"
DATASET = FINAL_DIR / "localnewsqa_targetqa_explicit_strict_defensible_1000_per_country_curated_final.jsonl"
CSV_OUT = FINAL_DIR / "localnewsqa_targetqa_explicit_strict_defensible_1000_per_country_curated_final.csv"
FETCH_CACHE = FINAL_DIR / "strict_defensible_1000_curated_final_target_evidence_fetches.jsonl"
SUMMARY = FINAL_DIR / "single_cue_repair_summary.json"
REPAIR_LOG = FINAL_DIR / "single_cue_repair_log.csv"
AMBIGUOUS = BASE / "localnewsqa_ambiguous_semantic_gold_1700.jsonl"

AUDIT_EXPLICIT_SCRIPT = ROOT / "qa_data/localnewsqa_core/48_audit_explicit_max_split.py"
FETCH_AUDIT_SCRIPT = ROOT / "qa_data/localnewsqa_core/32_web_audit_ambiguous_verifiable.py"
BUILDER_SCRIPT = ROOT / "qa_data/localnewsqa_core/45_build_relation_strict_gold_ambiguous.py"

TARGET_SOURCE_ID = "localnewsqa_explicit_0000765"
REPAIR = {
    "target_evidence_url": "https://en.wikipedia.org/wiki/2015_NHL_Winter_Classic",
    "target_evidence_title": "2015 NHL Winter Classic",
    "target_evidence_excerpt": (
        "The 2015 NHL Winter Classic was an outdoor regular season National Hockey League "
        "game played at Nationals Park in Washington, D.C., United States."
    ),
}

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


def norm(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_wikipedia_title(title: str) -> dict:
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "prop": "extracts|info",
            "explaintext": 1,
            "exintro": 0,
            "redirects": 1,
            "inprop": "url",
            "titles": title,
        },
        timeout=25,
        headers={"User-Agent": "LocalNewsQA-single-cue-repair/1.0"},
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    page = next(iter(pages.values()))
    text = re.sub(r"\s+", " ", str(page.get("extract", "") or "")).strip()
    return {
        "url": REPAIR["target_evidence_url"],
        "ok": bool(text),
        "status_code": "200" if text else "404",
        "final_url": page.get("fullurl", REPAIR["target_evidence_url"]),
        "content_type": "application/json",
        "title": page.get("title", title),
        "text": text[:120_000],
        "text_len": len(text[:120_000]),
        "error": "" if text else "wikipedia_api_empty_extract",
        "elapsed_sec": 0.0,
    }


def main() -> None:
    explicit_audit = load_module(AUDIT_EXPLICIT_SCRIPT, "audit48")
    fetch_audit = load_module(FETCH_AUDIT_SCRIPT, "fetch32")
    builder = load_module(BUILDER_SCRIPT, "builder45")

    rows = read_jsonl(DATASET)
    fetches = explicit_audit.load_fetch_cache(FETCH_CACHE)
    ambiguous_ids = {row["source_row_id"] for row in read_jsonl(AMBIGUOUS)}
    fetches[REPAIR["target_evidence_url"]] = fetch_wikipedia_title(REPAIR["target_evidence_title"])

    repair_log = []
    repaired_count = 0
    for row in rows:
        if row.get("source_row_id") != TARGET_SOURCE_ID:
            continue
        old = {
            "source_row_id": row.get("source_row_id", ""),
            "country": row.get("country", ""),
            "question": row.get("question", ""),
            "target_answer": row.get("target_answer", ""),
            "old_target_evidence_title": row.get("target_evidence_title", ""),
            "old_target_evidence_url": row.get("target_evidence_url", ""),
            "old_target_evidence_excerpt": row.get("target_evidence_excerpt", ""),
            "new_target_evidence_title": REPAIR["target_evidence_title"],
            "new_target_evidence_url": REPAIR["target_evidence_url"],
            "new_target_evidence_excerpt": REPAIR["target_evidence_excerpt"],
        }
        row.update(REPAIR)
        row["evidence_hint"] = REPAIR["target_evidence_excerpt"]
        row["curation_status"] = "manual_single_cue_evidence_repair_wikipedia_supported"
        row["curation_support_sentence"] = REPAIR["target_evidence_excerpt"]
        repair_log.append(old)
        repaired_count += 1

    failures = []
    warnings = []
    high_conf_rows = []
    anchor_rows = []
    for row in rows:
        if row.get("source_row_id") != TARGET_SOURCE_ID:
            continue
        row_failures, row_warnings = explicit_audit.audit_explicit_row(row, fetches, ambiguous_ids, fetch_audit)
        if row_failures:
            failures.append({"source_row_id": row.get("source_row_id", ""), "failures": " | ".join(row_failures)})
        if row_warnings:
            warnings.append({"source_row_id": row.get("source_row_id", ""), "warnings": " | ".join(row_warnings)})
        cue_text = builder.evidence_cue_text(row, "target", fetches)
        missing = [cue for cue in builder.high_confidence_question_cues(row) if cue["norm"] not in cue_text]
        if missing:
            high_conf_rows.append(
                {
                    "source_row_id": row.get("source_row_id", ""),
                    "country": row.get("country", ""),
                    "question": row.get("question", ""),
                    "missing_cues": " | ".join(cue["cue"] for cue in missing),
                }
            )
        fetch = fetches.get(row.get("target_evidence_url", ""), {})
        anchor_text = " ".join(
            str(part or "")
            for part in [
                row.get("target_evidence_title", ""),
                row.get("target_evidence_excerpt", ""),
                fetch.get("title", ""),
                str(fetch.get("text", ""))[:1500],
            ]
        )
        if not fetch_audit.contains_any(fetch_audit.normalize_text(anchor_text), fetch_audit.answer_aliases(row.get("target_answer", ""))):
            anchor_rows.append({"source_row_id": row.get("source_row_id", ""), "target_answer": row.get("target_answer", "")})

    rows.sort(
        key=lambda row: (
            COUNTRY_ORDER.index(row["country"]),
            norm(row.get("target_answer", "")),
            norm(row.get("question", "")),
            row.get("source_row_id", ""),
        )
    )
    for idx, row in enumerate(rows, start=1):
        row["id"] = f"localnewsqa_explicit_strict_defensible_1000_{idx:05d}"
        row["split_name"] = "LocalNewsQA-Explicit-Strict-Defensible-1000-Curated-Final"

    source_ids = [row["source_row_id"] for row in rows]
    question_keys = [norm(row["question"]) for row in rows]
    selected_urls = {row.get("target_evidence_url", "") for row in rows if row.get("target_evidence_url", "")}
    validation_errors = []
    if repaired_count != 1:
        validation_errors.append(f"expected 1 repaired row, got {repaired_count}")
    if len(rows) != 17_000:
        validation_errors.append(f"expected 17000 rows, got {len(rows)}")
    for country, count in Counter(row["country"] for row in rows).items():
        if count != 1000:
            validation_errors.append(f"{country}: expected 1000, got {count}")
    if len(source_ids) != len(set(source_ids)):
        validation_errors.append("duplicate source ids")
    if len(question_keys) != len(set(question_keys)):
        validation_errors.append("duplicate questions")
    if set(source_ids) & ambiguous_ids:
        validation_errors.append(f"ambiguous overlap: {len(set(source_ids) & ambiguous_ids)}")
    if failures:
        validation_errors.append(f"failure rows: {len(failures)}")
    if warnings:
        validation_errors.append(f"warning rows: {len(warnings)}")
    if high_conf_rows:
        validation_errors.append(f"high confidence cue rows: {len(high_conf_rows)}")
    if anchor_rows:
        validation_errors.append(f"answer anchor rows: {len(anchor_rows)}")

    write_jsonl(DATASET, rows)
    write_csv(CSV_OUT, rows)
    write_jsonl(FETCH_CACHE, [fetches[url] for url in sorted(selected_urls) if url in fetches])
    write_csv(REPAIR_LOG, repair_log)

    summary = {
        "rows": len(rows),
        "repaired_count": repaired_count,
        "failure_rows": len(failures),
        "warning_rows": len(warnings),
        "high_confidence_cue_rows": len(high_conf_rows),
        "answer_anchor_rows": len(anchor_rows),
        "country_counts": dict(Counter(row["country"] for row in rows)),
        "duplicate_source_ids": len(source_ids) - len(set(source_ids)),
        "duplicate_questions": len(question_keys) - len(set(question_keys)),
        "ambiguous_overlap": len(set(source_ids) & ambiguous_ids),
        "valid": not validation_errors,
        "validation_errors": validation_errors,
        "paths": {
            "jsonl": str(DATASET),
            "csv": str(CSV_OUT),
            "fetch_cache": str(FETCH_CACHE),
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
