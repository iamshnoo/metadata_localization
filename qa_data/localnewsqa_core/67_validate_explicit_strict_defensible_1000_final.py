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
BASE = ROOT / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_semantic_gold_ambiguous_1700"
FINAL_DIR = BASE / "explicit_max_audit/strict_defensible_1000_curated_final"
DEFAULT_INPUT = FINAL_DIR / "localnewsqa_targetqa_explicit_strict_defensible_1000_per_country_curated_final.jsonl"
DEFAULT_AMBIGUOUS = BASE / "localnewsqa_ambiguous_semantic_gold_1700.jsonl"
DEFAULT_OUTDIR = FINAL_DIR / "final_validation"
FETCH_CACHE = FINAL_DIR / "strict_defensible_1000_curated_final_target_evidence_fetches.jsonl"

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


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def norm(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def cue_tokens(question: str, answer: str, country: str) -> list[str]:
    stopwords = {
        "about",
        "after",
        "also",
        "annual",
        "before",
        "being",
        "between",
        "country",
        "during",
        "from",
        "government",
        "local",
        "major",
        "market",
        "mentioned",
        "national",
        "news",
        "often",
        "public",
        "reports",
        "story",
        "their",
        "this",
        "which",
        "with",
        "would",
    }
    answer_tokens = set(norm(answer).split())
    country_tokens = set(norm(country).split())
    cues = []
    for token in norm(question).split():
        if len(token) < 4 or token in stopwords or token in answer_tokens or token in country_tokens:
            continue
        if token.isdigit():
            continue
        cues.append(token)
    return sorted(set(cues))


def evidence_text(row: dict, fetch_norms: dict[str, str]) -> str:
    return norm(
        " ".join(
            str(part or "")
            for part in [
                row.get("target_evidence_url", ""),
                row.get("target_evidence_title", ""),
                row.get("target_evidence_excerpt", ""),
            ]
        )
    ) + " " + fetch_norms.get(row.get("target_evidence_url", ""), "")


def anchor_text(row: dict, fetches: dict[str, dict]) -> str:
    fetch = fetches.get(row.get("target_evidence_url", ""), {})
    return " ".join(
        str(part or "")
        for part in [
            row.get("target_evidence_title", ""),
            row.get("target_evidence_excerpt", ""),
            fetch.get("title", ""),
            str(fetch.get("text", ""))[:1500],
            str(fetch.get("api_text_for_curation", ""))[:1500],
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Final strict validation for the 17k explicit split.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--ambiguous", type=Path, default=DEFAULT_AMBIGUOUS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--fetch-cache", type=Path, default=FETCH_CACHE)
    args = parser.parse_args()

    explicit_audit = load_module(AUDIT_EXPLICIT_SCRIPT, "audit48")
    fetch_audit = load_module(FETCH_AUDIT_SCRIPT, "fetch32")
    builder = load_module(BUILDER_SCRIPT, "builder45")

    rows = read_jsonl(args.input)
    ambiguous_ids = {row["source_row_id"] for row in read_jsonl(args.ambiguous)}
    fetches = explicit_audit.load_fetch_cache(args.fetch_cache)
    fetch_norms = {
        url: norm(
            " ".join(
                str(part or "")
                for part in [
                    fetch.get("title", ""),
                    fetch.get("text", ""),
                    fetch.get("markdown", ""),
                    fetch.get("api_text_for_curation", ""),
                ]
            )
        )
        for url, fetch in fetches.items()
    }
    fetch_cue_norms = {
        url: builder.cue_norm(
            " ".join(
                str(part or "")
                for part in [
                    fetch.get("title", ""),
                    fetch.get("text", ""),
                    fetch.get("markdown", ""),
                    fetch.get("api_text_for_curation", ""),
                ]
            )
        )
        for url, fetch in fetches.items()
    }
    args.outdir.mkdir(parents=True, exist_ok=True)

    source_ids = [row.get("source_row_id", "") for row in rows]
    duplicate_source_ids = {source_id for source_id, count in Counter(source_ids).items() if count > 1}
    question_keys = [norm(row.get("question", "")) for row in rows]
    duplicate_question_keys = {question for question, count in Counter(question_keys).items() if count > 1}

    audit_failure_rows = []
    audit_warning_rows = []
    high_conf_rows = []
    answer_anchor_rows = []
    relation_probe_rows = []
    manual_support_rows = []
    failure_counts = Counter()
    warning_counts = Counter()

    for row in rows:
        failures, warnings = explicit_audit.audit_explicit_row(row, fetches, ambiguous_ids, fetch_audit)
        if row.get("source_row_id", "") in duplicate_source_ids:
            failures.append("duplicate_source_id")
        if norm(row.get("question", "")) in duplicate_question_keys:
            failures.append("duplicate_question_text")

        for failure in failures:
            failure_counts[failure] += 1
        for warning in warnings:
            warning_counts[warning] += 1
        if failures:
            audit_failure_rows.append(
                {
                    "source_row_id": row.get("source_row_id", ""),
                    "country": row.get("country", ""),
                    "source_split_type": row.get("source_split_type", ""),
                    "curation_status": row.get("curation_status", ""),
                    "question": row.get("question", ""),
                    "target_answer": row.get("target_answer", ""),
                    "target_evidence_title": row.get("target_evidence_title", ""),
                    "target_evidence_url": row.get("target_evidence_url", ""),
                    "failures": " | ".join(sorted(set(failures))),
                    "warnings": " | ".join(sorted(set(warnings))),
                }
            )
        if warnings:
            audit_warning_rows.append(
                {
                    "source_row_id": row.get("source_row_id", ""),
                    "country": row.get("country", ""),
                    "source_split_type": row.get("source_split_type", ""),
                    "curation_status": row.get("curation_status", ""),
                    "question": row.get("question", ""),
                    "target_answer": row.get("target_answer", ""),
                    "target_evidence_title": row.get("target_evidence_title", ""),
                    "target_evidence_url": row.get("target_evidence_url", ""),
                    "warnings": " | ".join(sorted(set(warnings))),
                }
            )

        cue_text = (
            builder.cue_norm(
                " ".join(
                    str(part or "")
                    for part in [
                        row.get("target_evidence_title", ""),
                        row.get("target_evidence_excerpt", ""),
                    ]
                )
            )
            + " "
            + fetch_cue_norms.get(row.get("target_evidence_url", ""), "")
        )
        missing_cues = [cue for cue in builder.high_confidence_question_cues(row) if cue["norm"] not in cue_text]
        if missing_cues:
            high_conf_rows.append(
                {
                    "source_row_id": row.get("source_row_id", ""),
                    "country": row.get("country", ""),
                    "question": row.get("question", ""),
                    "target_answer": row.get("target_answer", ""),
                    "target_evidence_title": row.get("target_evidence_title", ""),
                    "target_evidence_url": row.get("target_evidence_url", ""),
                    "missing_cues": " | ".join(cue["cue"] for cue in missing_cues),
                }
            )

        anchor_norm = fetch_audit.normalize_text(anchor_text(row, fetches))
        aliases = fetch_audit.answer_aliases(row.get("target_answer", ""))
        if not fetch_audit.contains_any(anchor_norm, aliases):
            answer_anchor_rows.append(
                {
                    "source_row_id": row.get("source_row_id", ""),
                    "country": row.get("country", ""),
                    "question": row.get("question", ""),
                    "target_answer": row.get("target_answer", ""),
                    "target_evidence_title": row.get("target_evidence_title", ""),
                    "target_evidence_url": row.get("target_evidence_url", ""),
                }
            )

        text = evidence_text(row, fetch_norms)
        cues = cue_tokens(row.get("question", ""), row.get("target_answer", ""), row.get("country", ""))
        cue_hits = [cue for cue in cues if cue in text]
        year_tokens = sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", row.get("question", ""))))
        missing_years = [year for year in year_tokens if year not in text]
        if len(cues) >= 4 and len(cue_hits) < 2:
            relation_probe_rows.append(
                {
                    "source_row_id": row.get("source_row_id", ""),
                    "country": row.get("country", ""),
                    "issue": "diagnostic_low_question_relation_cue_overlap",
                    "cue_count": len(cues),
                    "cue_hits": " | ".join(cue_hits[:20]),
                    "question": row.get("question", ""),
                    "target_answer": row.get("target_answer", ""),
                    "target_evidence_title": row.get("target_evidence_title", ""),
                }
            )
        if missing_years:
            relation_probe_rows.append(
                {
                    "source_row_id": row.get("source_row_id", ""),
                    "country": row.get("country", ""),
                    "issue": "diagnostic_temporal_year_not_supported",
                    "missing_years": " | ".join(missing_years),
                    "question": row.get("question", ""),
                    "target_answer": row.get("target_answer", ""),
                    "target_evidence_title": row.get("target_evidence_title", ""),
                }
            )

        if row.get("curation_status"):
            excerpt_norm = fetch_audit.normalize_text(row.get("target_evidence_excerpt", ""))
            manual_ok = fetch_audit.contains_any(excerpt_norm, aliases) and fetch_audit.contains_any(
                excerpt_norm,
                fetch_audit.marker_aliases(row.get("country", "")),
            )
            if not manual_ok:
                manual_support_rows.append(
                    {
                        "source_row_id": row.get("source_row_id", ""),
                        "country": row.get("country", ""),
                        "curation_status": row.get("curation_status", ""),
                        "question": row.get("question", ""),
                        "target_answer": row.get("target_answer", ""),
                        "target_evidence_title": row.get("target_evidence_title", ""),
                        "target_evidence_excerpt": row.get("target_evidence_excerpt", ""),
                    }
                )

    target_urls = {row.get("target_evidence_url", "") for row in rows if row.get("target_evidence_url", "")}
    country_counts = Counter(row.get("country", "") for row in rows)
    validation_errors = []
    if len(rows) != 17_000:
        validation_errors.append(f"expected 17000 rows, got {len(rows)}")
    for country in COUNTRY_ORDER:
        if country_counts[country] != 1000:
            validation_errors.append(f"{country}: expected 1000, got {country_counts[country]}")
    if any(country not in COUNTRY_ORDER for country in country_counts):
        validation_errors.append("unexpected countries present")
    if duplicate_source_ids:
        validation_errors.append(f"duplicate source ids: {len(duplicate_source_ids)}")
    if duplicate_question_keys:
        validation_errors.append(f"duplicate normalized questions: {len(duplicate_question_keys)}")
    overlap = set(source_ids) & ambiguous_ids
    if overlap:
        validation_errors.append(f"ambiguous source overlap: {len(overlap)}")
    if len(target_urls) != sum(1 for url in target_urls if url in fetches):
        validation_errors.append("target evidence fetch cache incomplete")
    if audit_failure_rows:
        validation_errors.append(f"audit failure rows: {len(audit_failure_rows)}")
    if audit_warning_rows:
        validation_errors.append(f"audit warning rows: {len(audit_warning_rows)}")
    if high_conf_rows:
        validation_errors.append(f"high confidence cue issue rows: {len(high_conf_rows)}")
    if answer_anchor_rows:
        validation_errors.append(f"answer anchor issue rows: {len(answer_anchor_rows)}")
    if manual_support_rows:
        validation_errors.append(f"manual curation excerpt support issue rows: {len(manual_support_rows)}")

    paths = {
        "audit_failures": args.outdir / "final_strict_explicit_audit_failures.csv",
        "audit_warnings": args.outdir / "final_strict_explicit_audit_warnings.csv",
        "high_confidence_cues": args.outdir / "final_strict_explicit_high_confidence_cue_issues.csv",
        "answer_anchor": args.outdir / "final_strict_explicit_answer_anchor_issues.csv",
        "relation_probe": args.outdir / "final_strict_explicit_relation_probe_diagnostics.csv",
        "manual_support": args.outdir / "final_strict_explicit_manual_support_issues.csv",
    }
    write_csv(paths["audit_failures"], audit_failure_rows)
    write_csv(paths["audit_warnings"], audit_warning_rows)
    write_csv(paths["high_confidence_cues"], high_conf_rows)
    write_csv(paths["answer_anchor"], answer_anchor_rows)
    write_csv(paths["relation_probe"], relation_probe_rows)
    write_csv(paths["manual_support"], manual_support_rows)

    summary_path = args.outdir / "final_strict_explicit_validation_summary.json"
    summary = {
        "input": str(args.input),
        "rows": len(rows),
        "country_counts": dict(country_counts),
        "source_split_type_counts": dict(Counter(row.get("source_split_type", "") for row in rows)),
        "curation_status_counts": dict(Counter(row.get("curation_status", "") for row in rows)),
        "target_unique_urls": len(target_urls),
        "target_fetch_coverage": sum(1 for url in target_urls if url in fetches),
        "target_fetch_ok": sum(1 for url in target_urls if fetches.get(url, {}).get("ok")),
        "duplicate_source_ids": len(duplicate_source_ids),
        "duplicate_normalized_questions": len(duplicate_question_keys),
        "ambiguous_overlap": len(overlap),
        "audit_failure_rows": len(audit_failure_rows),
        "audit_warning_rows": len(audit_warning_rows),
        "audit_failure_counts": dict(failure_counts),
        "audit_warning_counts": dict(warning_counts),
        "high_confidence_cue_issue_rows": len(high_conf_rows),
        "answer_anchor_issue_rows": len(answer_anchor_rows),
        "manual_curation_excerpt_support_issue_rows": len(manual_support_rows),
        "diagnostic_relation_probe_rows": len(relation_probe_rows),
        "valid": not validation_errors,
        "validation_errors": validation_errors,
        "paths": {key: str(path) for key, path in paths.items()} | {"summary": str(summary_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if validation_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
