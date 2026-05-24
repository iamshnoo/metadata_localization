#!/usr/bin/env python3

import argparse
import importlib.util
import json
from collections import Counter
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


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate final semantic-gold LocalNewsQA splits.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_DIR)
    parser.add_argument(
        "--ambiguous",
        type=Path,
        default=DEFAULT_DIR / "localnewsqa_ambiguous_semantic_gold_1700.jsonl",
    )
    parser.add_argument(
        "--explicit",
        type=Path,
        default=DEFAULT_DIR / "localnewsqa_targetqa_explicit_style_max_target_supported.jsonl",
    )
    parser.add_argument(
        "--fetches",
        type=Path,
        default=DEFAULT_DIR / "semantic_gold_selected_evidence_fetches.jsonl",
    )
    args = parser.parse_args()

    audit_mod = load_module(AUDIT_SCRIPT, "audit32")
    gold_mod = load_module(GOLD_SCRIPT, "gold41")
    repair_mod = load_module(REPAIR_SCRIPT, "repair43")
    ambiguous = read_jsonl(args.ambiguous)
    explicit = read_jsonl(args.explicit)
    fetch_rows = read_jsonl(args.fetches)
    fetches = {row["url"]: row for row in fetch_rows}

    audits = [audit_mod.audit_row(row, fetches) for row in ambiguous]
    semantic_risks = [
        {
            "source_row_id": row["source_row_id"],
            "country": row["country"],
            "risks": gold_mod.side_risk(row, "target", fetches)
            + gold_mod.side_risk(row, "contrast", fetches),
        }
        for row in ambiguous
    ]
    semantic_risks = [row for row in semantic_risks if row["risks"]]

    ambiguous_source_ids = [row["source_row_id"] for row in ambiguous]
    explicit_source_ids = [row["source_row_id"] for row in explicit]
    target_contrast_same_answer = [
        row["source_row_id"]
        for row in ambiguous
        if audit_mod.normalize_text(row.get("target_answer")) == audit_mod.normalize_text(row.get("contrast_answer"))
    ]
    target_contrast_same_country = [
        row["source_row_id"] for row in ambiguous if row.get("country") == row.get("contrast_country")
    ]
    target_contrast_same_evidence = [
        row["source_row_id"]
        for row in ambiguous
        if row.get("target_evidence_url") == row.get("contrast_evidence_url")
    ]
    option_shape_errors = []
    answer_membership_errors = []
    metadata_errors = []
    question_quality_errors = []
    weak_locale_errors = []
    for row in ambiguous:
        options = audit_mod.option_parts(row.get("options", []))
        norm_options = [audit_mod.normalize_text(option) for option in options]
        if len(options) != 4 or len(norm_options) != len(set(norm_options)):
            option_shape_errors.append(row["source_row_id"])
        if audit_mod.answer_count(options, row.get("target_answer", "")) != 1:
            answer_membership_errors.append(f"{row['source_row_id']}: target answer not exactly once")
        if audit_mod.answer_count(options, row.get("contrast_answer", "")) != 1:
            answer_membership_errors.append(f"{row['source_row_id']}: contrast answer not exactly once")
        if audit_mod.normalize_text(row.get("correct_answer")) != audit_mod.normalize_text(row.get("target_answer")):
            metadata_errors.append(f"{row['source_row_id']}: correct_answer != target_answer")
        if row.get("target_country") != row.get("country"):
            metadata_errors.append(f"{row['source_row_id']}: target_country != country")
        if row.get("split_type") != "ambiguous" or row.get("ambiguity_flag") is not True:
            metadata_errors.append(f"{row['source_row_id']}: split metadata not ambiguous")
        if row.get("weak_locale_risk") != "clean" or row.get("weak_locale_flags"):
            weak_locale_errors.append(row["source_row_id"])
        quality_issues = repair_mod.question_quality_issues(row.get("question", ""))
        if quality_issues:
            question_quality_errors.append(
                {
                    "source_row_id": row["source_row_id"],
                    "question": row.get("question", ""),
                    "issues": quality_issues,
                }
            )

    validation_errors = []
    country_counts = Counter(row["country"] for row in ambiguous)
    if len(ambiguous) != 1700:
        validation_errors.append(f"ambiguous row count is {len(ambiguous)}, expected 1700")
    for country, count in sorted(country_counts.items()):
        if count != 100:
            validation_errors.append(f"{country} has {count} ambiguous rows, expected 100")
    if len(country_counts) != 17:
        validation_errors.append(f"ambiguous country count is {len(country_counts)}, expected 17")
    if len(ambiguous_source_ids) != len(set(ambiguous_source_ids)):
        validation_errors.append("duplicate ambiguous source_row_id")
    if Counter(audit["severity"] for audit in audits) != {"pass": 1700}:
        validation_errors.append("ambiguous web audit is not all pass")
    if any(audit["warnings"] for audit in audits):
        validation_errors.append("ambiguous web audit has warnings")
    if semantic_risks:
        validation_errors.append(f"semantic evidence risk rows: {len(semantic_risks)}")
    if option_shape_errors:
        validation_errors.append(f"option shape/duplicate errors: {len(option_shape_errors)}")
    if answer_membership_errors:
        validation_errors.append(f"answer membership errors: {len(answer_membership_errors)}")
    if metadata_errors:
        validation_errors.append(f"metadata errors: {len(metadata_errors)}")
    if weak_locale_errors:
        validation_errors.append(f"weak-locale flagged rows: {len(weak_locale_errors)}")
    if question_quality_errors:
        validation_errors.append(f"question quality artifact rows: {len(question_quality_errors)}")
    if target_contrast_same_answer:
        validation_errors.append(f"same target/contrast answer rows: {len(target_contrast_same_answer)}")
    if target_contrast_same_country:
        validation_errors.append(f"same target/contrast country rows: {len(target_contrast_same_country)}")
    if target_contrast_same_evidence:
        validation_errors.append(f"same target/contrast evidence rows: {len(target_contrast_same_evidence)}")
    if len(explicit_source_ids) != len(set(explicit_source_ids)):
        validation_errors.append("duplicate explicit source_row_id")
    overlap = sorted(set(ambiguous_source_ids) & set(explicit_source_ids))
    if overlap:
        validation_errors.append(f"explicit/ambiguous overlap rows: {len(overlap)}")

    summary = {
        "ambiguous_path": str(args.ambiguous),
        "explicit_path": str(args.explicit),
        "selected_evidence_fetches_path": str(args.fetches),
        "ambiguous_rows": len(ambiguous),
        "explicit_rows": len(explicit),
        "country_counts": dict(sorted(country_counts.items())),
        "ambiguous_web_audit_severity_counts": dict(Counter(audit["severity"] for audit in audits)),
        "ambiguous_web_audit_warning_counts": dict(Counter(audit["warnings"] for audit in audits)),
        "ambiguous_web_audit_failure_counts": dict(Counter(audit["failures"] for audit in audits if audit["failures"])),
        "semantic_evidence_risk_rows": len(semantic_risks),
        "duplicate_ambiguous_source_ids": len(ambiguous_source_ids) - len(set(ambiguous_source_ids)),
        "duplicate_explicit_source_ids": len(explicit_source_ids) - len(set(explicit_source_ids)),
        "explicit_ambiguous_overlap_rows": len(overlap),
        "same_target_contrast_answer_rows": len(target_contrast_same_answer),
        "same_target_contrast_country_rows": len(target_contrast_same_country),
        "same_target_contrast_evidence_url_rows": len(target_contrast_same_evidence),
        "option_shape_error_rows": len(option_shape_errors),
        "answer_membership_error_rows": len(answer_membership_errors),
        "metadata_error_rows": len(metadata_errors),
        "weak_locale_error_rows": len(weak_locale_errors),
        "question_quality_artifact_rows": len(question_quality_errors),
        "question_repair_applied_rows": sum(1 for row in ambiguous if row.get("semantic_gold_question_repair_applied") is True),
        "ambiguous_source_split_type_counts": dict(Counter(row.get("source_split_type", "") for row in ambiguous)),
        "explicit_source_split_type_counts": dict(Counter(row.get("source_split_type", "") for row in explicit)),
        "ambiguous_weak_locale_risk_counts": dict(Counter(row.get("weak_locale_risk", "") for row in ambiguous)),
        "ambiguous_weak_locale_flag_counts": dict(Counter(row.get("weak_locale_flags", "") for row in ambiguous)),
        "ambiguous_semantic_risk_band_counts": dict(Counter(row.get("semantic_risk_band", "") for row in ambiguous)),
        "selected_evidence_fetch_count": len(fetches),
        "validation_errors": validation_errors,
        "valid": not validation_errors,
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / "semantic_gold_final_validation_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if validation_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
