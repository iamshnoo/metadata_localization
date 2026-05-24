#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_semantic_gold_ambiguous_1700"
)
AUDIT_SCRIPT = ROOT / "qa_data/localnewsqa_core/32_web_audit_ambiguous_verifiable.py"
GOLD_SCRIPT = ROOT / "qa_data/localnewsqa_core/41_build_semantic_gold_ambiguous.py"
REPAIR_SCRIPT = ROOT / "qa_data/localnewsqa_core/43_repair_semantic_gold_questions.py"
RELATION_SCRIPT = ROOT / "qa_data/localnewsqa_core/44_relation_support_audit.py"
REVIEWER_SCRIPT = ROOT / "qa_data/localnewsqa_core/47_reviewer_risk_audit.py"
WEAK_SCRIPT = ROOT / "qa_data/localnewsqa_core/26_flag_weak_locale_ambiguous.py"


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
    fields: list[str] = []
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


BAD_QUESTION_PATTERNS = {
    "placeholder_relevant_reference": r"\brelevant reference\b",
    "placeholder_relevant_public_figure": r"\brelevant public figure\b",
    "placeholder_relevant_organization": r"\brelevant organization\b",
    "placeholder_relevant_public_body": r"\brelevant public body\b",
    "fallback_matches_local_reporting": r"\bmatches local reporting about\b",
    "fallback_fits_local_context": r"\bfits (?:the )?(?:described )?(?:local )?context\b",
    "fallback_which_answer": r"^which (?:answer|option)\b",
    "awkward_late_relevant_year": r"\blate[- ]the relevant year\b",
    "awkward_question_ends_by": r"\bby\?$",
    "awkward_lowercase_st": r"\bst [a-z]",
    "incomplete_trailing_word": r"\b(?:is|are|was|were|the|of|in|at|by)\??$",
    "double_period": r"\.\?",
}


def bad_question_hits(question: str, repair_mod: Any) -> list[str]:
    hits = repair_mod.question_quality_issues(question)
    for name, pattern in BAD_QUESTION_PATTERNS.items():
        if re.search(pattern, question, flags=re.IGNORECASE):
            hits.append(name)
    return sorted(set(hits))


def main() -> None:
    parser = argparse.ArgumentParser(description="Final strict quality audit for LocalNewsQA semantic-gold ambiguous split.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--input", type=Path, default=DEFAULT_DIR / "localnewsqa_ambiguous_semantic_gold_1700.jsonl")
    parser.add_argument("--fetches", type=Path, default=DEFAULT_DIR / "semantic_gold_selected_evidence_fetches.jsonl")
    args = parser.parse_args()

    audit_mod = load_module(AUDIT_SCRIPT, "audit32")
    gold_mod = load_module(GOLD_SCRIPT, "gold41")
    repair_mod = load_module(REPAIR_SCRIPT, "repair43")
    relation_mod = load_module(RELATION_SCRIPT, "relation44")
    reviewer_mod = load_module(REVIEWER_SCRIPT, "reviewer47")
    weak_mod = load_module(WEAK_SCRIPT, "weak26")

    rows = read_jsonl(args.input)
    fetches = {row["url"]: row for row in read_jsonl(args.fetches)}

    failures = []
    side_audits = []
    row_web_counts = Counter()
    raw_question_issue_counts = Counter()
    reviewer_risk_counts = Counter()
    country_counts = Counter(row.get("country", "") for row in rows)

    if len(rows) != 1700:
        failures.append({"scope": "file", "issue": "row_count", "detail": str(len(rows))})
    for country in gold_mod.COUNTRY_ORDER:
        if country_counts[country] != 100:
            failures.append({"scope": "file", "issue": "country_count", "detail": f"{country}:{country_counts[country]}"})

    source_ids = [row.get("source_row_id", "") for row in rows]
    for source_id, count in Counter(source_ids).items():
        if count > 1:
            failures.append({"scope": "row", "source_row_id": source_id, "issue": "duplicate_source_id", "detail": str(count)})

    for row in rows:
        source_id = row.get("source_row_id", "")
        options = audit_mod.option_parts(row.get("options", []))
        norm_options = [audit_mod.normalize_text(option) for option in options]
        row_checks = []
        if len(options) != 4:
            row_checks.append("option_count_not_4")
        if len(norm_options) != len(set(norm_options)):
            row_checks.append("duplicate_normalized_options")
        if audit_mod.answer_count(options, row.get("target_answer", "")) != 1:
            row_checks.append("target_answer_not_exactly_once")
        if audit_mod.answer_count(options, row.get("contrast_answer", "")) != 1:
            row_checks.append("contrast_answer_not_exactly_once")
        if audit_mod.normalize_text(row.get("target_answer")) == audit_mod.normalize_text(row.get("contrast_answer")):
            row_checks.append("same_target_contrast_answer")
        if row.get("country") == row.get("contrast_country"):
            row_checks.append("same_country_pair")
        if row.get("target_evidence_url") == row.get("contrast_evidence_url"):
            row_checks.append("same_evidence_url")
        if row.get("weak_locale_risk") != "clean" or row.get("weak_locale_flags"):
            row_checks.append("weak_locale_not_clean_after_adjudication")
        if row.get("web_audit_severity") != "pass":
            row_checks.append("stored_web_audit_not_pass")
        if row.get("relation_support_status") != "pass":
            row_checks.append("stored_relation_not_pass")
        for side in ["target", "contrast"]:
            if row.get(f"{side}_evidence_url") not in fetches:
                row_checks.append(f"{side}_fetch_missing")
        question_issues = bad_question_hits(row.get("question", ""), repair_mod)
        raw_question_issue_counts.update(question_issues)
        row_checks.extend(f"question:{issue}" for issue in question_issues)

        web_audit = audit_mod.audit_row(row, fetches)
        row_web_counts[web_audit["severity"]] += 1
        if web_audit["severity"] != "pass":
            row_checks.append(f"rerun_web_audit_{web_audit['severity']}:{web_audit['failures'] or web_audit['warnings']}")
        for risk in gold_mod.side_risk(row, "target", fetches) + gold_mod.side_risk(row, "contrast", fetches):
            row_checks.append(f"evidence_risk:{risk}")
        reviewer_risks = reviewer_mod.reviewer_risks(row, fetches, audit_mod, weak_mod)
        reviewer_risk_counts.update(risk["risk"] for risk in reviewer_risks)
        for risk in reviewer_risks:
            row_checks.append(f"reviewer_visible_cue_risk:{risk['risk']}:{risk['cue']}")

        for side in ["target", "contrast"]:
            side_audit = relation_mod.side_relation_audit(row, side, fetches, audit_mod)
            side_audits.append(side_audit)
            if side_audit["severity"] != "pass":
                row_checks.append(f"{side}_relation_{side_audit['severity']}:{side_audit['issues']}")

        for check in row_checks:
            failures.append(
                {
                    "scope": "row",
                    "source_row_id": source_id,
                    "country": row.get("country", ""),
                    "question": row.get("question", ""),
                    "target_answer": row.get("target_answer", ""),
                    "contrast_answer": row.get("contrast_answer", ""),
                    "issue": check,
                }
            )

    side_counts = Counter(row["severity"] for row in side_audits)
    issue_counts = Counter(issue for row in side_audits for issue in row["issues"].split(" | ") if issue)
    adjudication_counts = Counter(row.get("semantic_gold_question_adjudication", "") for row in rows)
    exact_question_counts = Counter(row.get("question", "") for row in rows)
    duplicate_questions = {question: count for question, count in exact_question_counts.items() if count > 1}
    if duplicate_questions:
        for row in rows:
            question = row.get("question", "")
            if question in duplicate_questions:
                failures.append(
                    {
                        "scope": "row",
                        "source_row_id": row.get("source_row_id", ""),
                        "country": row.get("country", ""),
                        "question": question,
                        "target_answer": row.get("target_answer", ""),
                        "contrast_answer": row.get("contrast_answer", ""),
                        "issue": f"duplicate_question:{duplicate_questions[question]}",
                    }
                )

    failures_path = args.outdir / "semantic_gold_final_quality_failures.csv"
    side_audit_path = args.outdir / "semantic_gold_final_relation_side_audit.csv"
    summary_path = args.outdir / "semantic_gold_final_quality_summary.json"
    write_csv(failures_path, failures)
    write_csv(side_audit_path, side_audits)
    summary = {
        "input": str(args.input),
        "fetches": str(args.fetches),
        "rows": len(rows),
        "country_counts": dict(country_counts),
        "valid": not failures,
        "failure_count": len(failures),
        "row_web_rerun_counts": dict(row_web_counts),
        "relation_side_counts": dict(side_counts),
        "relation_issue_counts": dict(issue_counts),
        "strong_relation_side_count": sum(1 for row in side_audits if row["strong_relation"]),
        "question_issue_counts": dict(raw_question_issue_counts),
        "reviewer_visible_cue_risk_counts": dict(reviewer_risk_counts),
        "reviewer_visible_cue_risk_count": sum(reviewer_risk_counts.values()),
        "question_adjudication_counts": dict(adjudication_counts),
        "duplicate_question_count": len(duplicate_questions),
        "duplicate_question_max_count": max(duplicate_questions.values()) if duplicate_questions else 0,
        "paths": {
            "failures": str(failures_path),
            "relation_side_audit": str(side_audit_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
