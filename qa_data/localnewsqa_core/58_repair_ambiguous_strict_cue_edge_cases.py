#!/usr/bin/env python3

import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515"
OUTDIR = BASE / "web_semantic_gold_ambiguous_1700"
INPUT_JSONL = OUTDIR / "localnewsqa_ambiguous_semantic_gold_1700.jsonl"
INPUT_CSV = OUTDIR / "localnewsqa_ambiguous_semantic_gold_1700.csv"
FETCHES = OUTDIR / "semantic_gold_selected_evidence_fetches.jsonl"
POOL = BASE / "localnewsqa_ambiguous_verifiable_pool_4625.jsonl"
SOURCE_FETCH_CACHE = BASE / "web_audit_ambiguous_1700/url_fetches.jsonl"
LOG_PATH = OUTDIR / "strict_cue_edge_case_repair_log.csv"
SUMMARY_PATH = OUTDIR / "strict_cue_edge_case_repair_summary.json"

AUDIT_SCRIPT = ROOT / "qa_data/localnewsqa_core/32_web_audit_ambiguous_verifiable.py"
GOLD_SCRIPT = ROOT / "qa_data/localnewsqa_core/41_build_semantic_gold_ambiguous.py"
REPAIR_SCRIPT = ROOT / "qa_data/localnewsqa_core/43_repair_semantic_gold_questions.py"
RELATION_SCRIPT = ROOT / "qa_data/localnewsqa_core/44_relation_support_audit.py"
WEAK_SCRIPT = ROOT / "qa_data/localnewsqa_core/26_flag_weak_locale_ambiguous.py"
REVIEWER_SCRIPT = ROOT / "qa_data/localnewsqa_core/47_reviewer_risk_audit.py"
BUILDER_SCRIPT = ROOT / "qa_data/localnewsqa_core/45_build_relation_strict_gold_ambiguous.py"


QUESTION_REPAIRS = {
    "localnewsqa_ambig_0024199": "Which business group is associated with retail, property, and media operations in this market?",
    "localnewsqa_ambig_0024769": "Which major port city is an important commercial hub on this country's southern coast?",
    "localnewsqa_ambig_0025769": "Which inland city is a major destination in the central part of the country?",
    "localnewsqa_ambig_0025273": "Which private television channel is part of a major media network in this market?",
    "localnewsqa_ambig_0025690": "Which hill-country city is the capital of its province or state?",
    "localnewsqa_ambig_0030685": "Which major city is served by the country's busiest airport?",
    "localnewsqa_ambig_0031341": "Which daily newspaper is a leading national newspaper in this market?",
    "localnewsqa_ambig_0031758": "Which inland city is a major urban centre in its region?",
    "localnewsqa_ambig_0031885": "Which town or city is an important administrative and commercial centre in its region?",
    "localnewsqa_ambig_0030878": "Who was the lower-house speaker in 2015?",
    "localnewsqa_ambig_0031763": "Which coastal town or city is known for beach tourism and seaside attractions?",
    "localnewsqa_ambig_0027874": "Which capital city is also the country's largest city and a main financial centre?",
}

ADD_SOURCE_IDS = {"localnewsqa_ambig_0027874"}


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
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def selected_fetch_rows(rows: list[dict], fetches: dict[str, dict]) -> list[dict]:
    urls = sorted(
        {
            row.get(f"{side}_evidence_url", "")
            for row in rows
            for side in ["target", "contrast"]
            if row.get(f"{side}_evidence_url", "")
        }
    )
    return [fetches[url] for url in urls if url in fetches]


def refresh_row_audit_fields(row: dict, fetches: dict[str, dict], mods: dict[str, Any]) -> tuple[dict, dict]:
    audit_mod = mods["audit"]
    relation_mod = mods["relation"]
    reviewer_mod = mods["reviewer"]
    weak_mod = mods["weak"]
    builder_mod = mods["builder"]
    gold_mod = mods["gold"]

    out = dict(row)
    old_question = str(out.get("question", ""))
    source_id = out["source_row_id"]
    if source_id in QUESTION_REPAIRS:
        out["question"] = QUESTION_REPAIRS[source_id]
        out["semantic_gold_question_repair_applied"] = True
        out["semantic_gold_question_adjudication"] = "manual_override_strict_cue"
        out["semantic_gold_question_repair_old_question"] = old_question
        out["evidence_hint"] = QUESTION_REPAIRS[source_id]

    web_audit = audit_mod.audit_row(out, fetches)
    warning_parts = [part.strip() for part in web_audit["warnings"].split(" | ") if part.strip()]
    out["semantic_gold_status"] = "pass"
    out["semantic_gold_notes"] = web_audit["warnings"]
    out["semantic_gold_mechanical_web_audit_severity"] = web_audit["severity"]
    out["semantic_gold_warning_count"] = len(warning_parts)
    out["web_audit_severity"] = "pass" if gold_mod.acceptable_audit(web_audit) else web_audit["severity"]
    out["web_audit_failures"] = "" if gold_mod.acceptable_audit(web_audit) else web_audit["failures"]
    out["web_audit_warnings"] = web_audit["warnings"]

    side_audits = [
        relation_mod.side_relation_audit(out, "target", fetches, audit_mod),
        relation_mod.side_relation_audit(out, "contrast", fetches, audit_mod),
    ]
    reviewer_risks = reviewer_mod.reviewer_risks(out, fetches, audit_mod, weak_mod)
    high_conf_risks = builder_mod.high_confidence_cue_issues(out, fetches)
    out["relation_support_status"] = "pass" if all(item["severity"] == "pass" for item in side_audits) else "fail"
    out["relation_target_strong"] = side_audits[0]["strong_relation"]
    out["relation_contrast_strong"] = side_audits[1]["strong_relation"]
    out["relation_target_cue_hits"] = side_audits[0]["cue_hits"]
    out["relation_contrast_cue_hits"] = side_audits[1]["cue_hits"]
    out["reviewer_visible_cue_risk_count"] = len(reviewer_risks)
    out["reviewer_visible_cue_risks"] = " | ".join(f"{risk['risk']}:{risk['cue']}" for risk in reviewer_risks)
    out["high_confidence_question_cue_risk_count"] = len(high_conf_risks)
    out["high_confidence_question_cue_risks"] = " | ".join(
        f"{risk['kind']}:{risk['cue']} missing={risk['missing_sides']}" for risk in high_conf_risks
    )

    status = {
        "source_row_id": source_id,
        "country": out.get("country", ""),
        "old_question": old_question,
        "new_question": out.get("question", ""),
        "web_severity": web_audit["severity"],
        "web_failures": web_audit["failures"],
        "web_warnings": web_audit["warnings"],
        "relation_severities": " | ".join(f"{item['side']}:{item['severity']}" for item in side_audits),
        "relation_issues": " || ".join(f"{item['side']}:{item['issues']}" for item in side_audits if item["issues"]),
        "reviewer_risk_count": len(reviewer_risks),
        "reviewer_risks": out["reviewer_visible_cue_risks"],
        "high_confidence_risk_count": len(high_conf_risks),
        "high_confidence_risks": out["high_confidence_question_cue_risks"],
    }
    return out, status


def main() -> None:
    mods = {
        "audit": load_module(AUDIT_SCRIPT, "audit32"),
        "gold": load_module(GOLD_SCRIPT, "gold41"),
        "repair": load_module(REPAIR_SCRIPT, "repair43"),
        "relation": load_module(RELATION_SCRIPT, "relation44"),
        "weak": load_module(WEAK_SCRIPT, "weak26"),
        "reviewer": load_module(REVIEWER_SCRIPT, "reviewer47"),
        "builder": load_module(BUILDER_SCRIPT, "builder45"),
    }

    fetches = mods["audit"].load_existing_fetches(SOURCE_FETCH_CACHE)
    fetches.update(mods["audit"].load_existing_fetches(FETCHES))
    rows = read_jsonl(INPUT_JSONL)
    pool_by_source = {row["source_row_id"]: row for row in read_jsonl(POOL)}
    existing_sources = {row["source_row_id"] for row in rows}

    for source_id in ADD_SOURCE_IDS:
        if source_id not in existing_sources:
            source = dict(pool_by_source[source_id])
            repaired_evidence, _ = mods["gold"].apply_overrides(source, {}, {}, fetches)
            repaired_question, _ = mods["repair"].repair_row(repaired_evidence, mods["weak"])
            rows.append(repaired_question)

    repaired_rows = []
    repair_log = []
    for row in rows:
        if row["source_row_id"] in QUESTION_REPAIRS:
            row, status = refresh_row_audit_fields(row, fetches, mods)
            repair_log.append(status)
        else:
            row, status = refresh_row_audit_fields(row, fetches, mods)
        repaired_rows.append(row)

    country_order = mods["gold"].COUNTRY_ORDER
    repaired_rows.sort(key=lambda row: (country_order.index(row["country"]), row["source_row_id"]))
    for idx, row in enumerate(repaired_rows, start=1):
        row["id"] = f"localnewsqa_ambig_semantic_gold_{idx:04d}"
        row["split_name"] = "LocalNewsQA-Ambiguous-Semantic-Gold-1700"

    validation_errors = mods["gold"].validate(repaired_rows)
    issue_rows = [
        row
        for row in repaired_rows
        if row.get("relation_support_status") != "pass"
        or int(row.get("reviewer_visible_cue_risk_count", 0) or 0) > 0
        or int(row.get("high_confidence_question_cue_risk_count", 0) or 0) > 0
        or row.get("web_audit_severity") != "pass"
    ]
    if issue_rows:
        validation_errors.append(
            "strict cue repair left issue rows: "
            + ", ".join(f"{row['source_row_id']}:{row.get('relation_support_status')}" for row in issue_rows[:20])
        )

    write_jsonl(INPUT_JSONL, repaired_rows)
    mods["gold"].write_csv(INPUT_CSV, repaired_rows)
    write_jsonl(FETCHES, selected_fetch_rows(repaired_rows, fetches))
    write_csv(LOG_PATH, repair_log)

    summary = {
        "rows": len(repaired_rows),
        "valid": not validation_errors,
        "validation_errors": validation_errors,
        "country_counts": dict(Counter(row["country"] for row in repaired_rows)),
        "added_rows": sorted(ADD_SOURCE_IDS),
        "question_repairs": len(repair_log),
        "reviewer_visible_cue_risk_rows": sum(
            1 for row in repaired_rows if int(row.get("reviewer_visible_cue_risk_count", 0) or 0) > 0
        ),
        "high_confidence_question_cue_risk_rows": sum(
            1 for row in repaired_rows if int(row.get("high_confidence_question_cue_risk_count", 0) or 0) > 0
        ),
        "relation_nonpass_rows": sum(1 for row in repaired_rows if row.get("relation_support_status") != "pass"),
        "web_nonpass_rows": sum(1 for row in repaired_rows if row.get("web_audit_severity") != "pass"),
        "duplicate_source_ids": len(repaired_rows) - len({row["source_row_id"] for row in repaired_rows}),
        "duplicate_questions": len(repaired_rows)
        - len({str(row.get("question", "")).strip().lower() for row in repaired_rows}),
        "paths": {
            "jsonl": str(INPUT_JSONL),
            "csv": str(INPUT_CSV),
            "fetches": str(FETCHES),
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
