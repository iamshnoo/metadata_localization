#!/usr/bin/env python3

import csv
import importlib.util
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_semantic_gold_ambiguous_1700"
)
AUDIT_DIR = BASE_DIR / "explicit_max_audit"
STRICT_DIR = AUDIT_DIR / "strict_1000"
SPLIT_JSONL = STRICT_DIR / "localnewsqa_targetqa_explicit_strict_1000_per_country_polished.jsonl"
SPLIT_CSV = STRICT_DIR / "localnewsqa_targetqa_explicit_strict_1000_per_country_polished.csv"
BROAD_45_CSV = STRICT_DIR / "remaining_broad_evidence_45.csv"
FETCH_CACHE = AUDIT_DIR / "explicit_target_evidence_fetches.jsonl"
OUT_DIR = STRICT_DIR / "remaining_broad_evidence_45_deep_audit"
FETCH_SCRIPT = ROOT / "qa_data/localnewsqa_core/32_web_audit_ambiguous_verifiable.py"

BROAD_TITLES = {"Education in Jamaica", "Public holidays in Jamaica"}

ASH = (
    "Ash Wednesday in Jamaica",
    "https://www.officeholidays.com/holidays/jamaica/ash-wednesday/",
    "Ash Wednesday in Jamaica: national holiday dates and 46-day relation to Easter.",
)
BOXING = (
    "Boxing Day in Jamaica",
    "https://www.officeholidays.com/holidays/jamaica/boxing-day/",
    "Boxing Day in Jamaica: national holiday observed after Christmas.",
)
CHRISTMAS = (
    "Christmas Day in Jamaica",
    "https://www.officeholidays.com/holidays/jamaica/christmas-day/",
    "Christmas Day in Jamaica: national holiday on December 25.",
)
GOOD_FRIDAY = (
    "Good Friday in Jamaica",
    "https://www.officeholidays.com/holidays/jamaica/good-friday/",
    "Good Friday in Jamaica: national holiday on the Friday before Easter.",
)
INDEPENDENCE = (
    "Independence Day (Jamaica)",
    "https://en.wikipedia.org/wiki/Independence_Day_(Jamaica)",
    "Independence Day in Jamaica: August 6 national day marking independence in 1962.",
)
LABOUR = (
    "Labour Day in Jamaica",
    "https://www.officeholidays.com/holidays/jamaica/jamaica-labour-day/",
    "Labour Day in Jamaica: May 23 national holiday linked to community labour projects.",
)
NEW_YEAR = (
    "New Year's Day in Jamaica",
    "https://time.bi/holidays/jamaica/new-years-day/",
    "New Year's Day in Jamaica: public holiday on January 1.",
)
MOEY_ABOUT = (
    "Jamaica Ministry of Education primary-level system",
    "https://moey.gov.jm/about/",
    "Jamaica Ministry of Education: primary education in Grades 1-6, GSAT, and preparatory schools.",
)
GOILP = (
    "Grade 1 Individual Learning Profile",
    "https://moey.gov.jm/grade-one-individual-learning-profile/",
    "Jamaica Ministry of Education: Grade One Individual Learning Profile for children entering Grade One.",
)

EVIDENCE_BY_ANSWER = {
    "Ash Wednesday": ASH,
    "Boxing Day": BOXING,
    "Christmas Day": CHRISTMAS,
    "Good Friday": GOOD_FRIDAY,
    "Independence": INDEPENDENCE,
    "Independence Day": INDEPENDENCE,
    "Labour Day": LABOUR,
    "New Year's Day": NEW_YEAR,
    "Grade 6": MOEY_ABOUT,
    "Preparatory schools": MOEY_ABOUT,
    "Grade 1": GOILP,
}

QUESTION_REWRITES = {
    "localnewsqa_explicit_0003252": (
        "Which Jamaican public holiday marks the start of Lent and can fall between February 4 and March 10?"
    ),
    "localnewsqa_explicit_0003201": (
        "Jamaica's early-August cultural celebration starts with Emancipation Day and ends with which national holiday?"
    ),
    "localnewsqa_explicit_0003212": (
        "In Jamaica, which national holiday is celebrated annually on the fixed date August 6?"
    ),
    "localnewsqa_explicit_0003210": (
        "Which Jamaican public holiday opens the calendar year on January 1?"
    ),
    "localnewsqa_ambig_0021384": (
        "In Jamaica, the Ministry of Education's Grade One Individual Learning Profile is used for children entering which grade?"
    ),
    "localnewsqa_ambig_0021677": (
        "In Jamaica, annual parades on the August 6 national day mark what occasion?"
    ),
    "localnewsqa_ambig_0021698": (
        "In Jamaica, which public holiday falls in late May and is tied to community labour projects?"
    ),
}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
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


def read_csv_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as handle:
        return {row["source_row_id"] for row in csv.DictReader(handle)}


def load_fetch_cache(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("url"):
                    out[row["url"]] = row
    return out


def write_fetch_cache(path: Path, fetches: dict[str, dict[str, Any]]) -> None:
    write_jsonl(path, [fetches[url] for url in sorted(fetches)])


def norm(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains(text_norm: str, phrase: str) -> bool:
    phrase_norm = norm(phrase)
    return bool(phrase_norm) and f" {phrase_norm} " in f" {text_norm} "


def contains_any(text_norm: str, phrases: list[str]) -> bool:
    return any(contains(text_norm, phrase) for phrase in phrases)


def contains_all(text_norm: str, phrases: list[str]) -> bool:
    return all(contains(text_norm, phrase) for phrase in phrases)


def evidence_blob(row: dict[str, Any], fetches: dict[str, dict[str, Any]]) -> str:
    fetch = fetches.get(row.get("target_evidence_url", ""), {})
    return " ".join(
        [
            row.get("target_evidence_title", ""),
            row.get("target_evidence_url", ""),
            row.get("target_evidence_excerpt", ""),
            fetch.get("title", ""),
            fetch.get("text", ""),
        ]
    )


def apply_repairs(rows: list[dict[str, Any]], broad_ids: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repaired = []
    log_rows = []
    for row in rows:
        working = dict(row)
        if working.get("source_row_id") in broad_ids:
            old_question = working.get("question", "")
            old_original_question = working.get("original_question", "")
            old_localized_question = working.get("localized_question", "")
            old_title = working.get("target_evidence_title", "")
            old_url = working.get("target_evidence_url", "")
            source_id = working.get("source_row_id", "")
            question_rewrite_applied = source_id in QUESTION_REWRITES
            if source_id in QUESTION_REWRITES:
                new_question = QUESTION_REWRITES[source_id]
                working["question"] = new_question
                working["original_question"] = new_question
                working["localized_question"] = new_question
            title, url, excerpt = EVIDENCE_BY_ANSWER[working["target_answer"]]
            working["target_evidence_title"] = title
            working["target_evidence_url"] = url
            working["target_evidence_excerpt"] = excerpt
            working["evidence_hint"] = excerpt
            working.pop("explicit_evidence_repair", None)
            log_rows.append(
                {
                    "source_row_id": source_id,
                    "target_answer": working.get("target_answer", ""),
                    "old_question": old_question,
                    "old_original_question": old_original_question,
                    "old_localized_question": old_localized_question,
                    "new_question": working.get("question", ""),
                    "question_rewrite_applied": question_rewrite_applied,
                    "question_changed": question_rewrite_applied
                    and any(
                        old != working.get("question", "")
                        for old in [old_question, old_original_question, old_localized_question]
                    ),
                    "old_target_evidence_title": old_title,
                    "old_target_evidence_url": old_url,
                    "new_target_evidence_title": title,
                    "new_target_evidence_url": url,
                }
            )
        repaired.append(working)
    return repaired, log_rows


def fetch_needed(urls: set[str], fetches: dict[str, dict[str, Any]], fetch_mod: Any) -> None:
    for url in sorted(urls):
        if fetches.get(url, {}).get("ok"):
            continue
        fetches[url] = fetch_mod.fetch_url(url, timeout=25.0, delay=0.2)
        time.sleep(0.2)


def option_shape_ok(row: dict[str, Any]) -> tuple[bool, list[str]]:
    issues = []
    options = row.get("options") or []
    if len(options) != 4:
        issues.append("option_count_not_4")
    normalized = [norm(option) for option in options]
    if len(normalized) != len(set(normalized)):
        issues.append("duplicate_normalized_options")
    if normalized.count(norm(row.get("target_answer", ""))) != 1:
        issues.append("target_answer_not_exactly_once")
    return not issues, issues


def relation_requirements(row: dict[str, Any]) -> list[tuple[str, list[str]]]:
    source_id = row.get("source_row_id", "")
    answer = row.get("target_answer", "")
    question = row.get("question", "")
    qn = norm(question)
    checks: list[tuple[str, list[str]]] = []

    if answer == "Ash Wednesday":
        checks.extend(
            [
                ("answer_name", ["ash wednesday"]),
                ("jamaica_page", ["jamaica"]),
                ("holiday_status", ["national holiday", "public holiday"]),
            ]
        )
        if "46" in qn:
            checks.append(("easter_offset", ["46 days before easter"]))
        if "start of lent" in qn:
            checks.append(("lent_start", ["start of lent", "beginning of lent"]))
            checks.append(("date_window_start", ["4 february", "february 4"]))
            checks.append(("date_window_end", ["march 10", "10 march"]))
    elif answer == "Boxing Day":
        checks.extend(
            [
                ("answer_name", ["boxing day"]),
                ("jamaica_page", ["jamaica"]),
                ("holiday_status", ["national holiday", "public holiday"]),
                ("december_26", ["dec 26", "december 26", "26 december", "26th"]),
            ]
        )
        if "follows christmas" in qn:
            checks.append(("after_christmas", ["day after christmas", "after christmas"]))
    elif answer == "Christmas Day":
        checks.extend(
            [
                ("answer_name", ["christmas day"]),
                ("jamaica_page", ["jamaica"]),
                ("holiday_status", ["national holiday", "public holiday"]),
                ("december_25", ["dec 25", "december 25", "25 december", "25th"]),
            ]
        )
    elif answer == "Good Friday":
        checks.extend(
            [
                ("answer_name", ["good friday"]),
                ("jamaica_page", ["jamaica"]),
                ("holiday_status", ["national holiday", "public holiday"]),
                ("friday_before_easter", ["friday before easter", "friday of holy week"]),
            ]
        )
    elif answer in {"Independence Day", "Independence"}:
        checks.extend(
            [
                ("answer_name", ["independence day", "independence"]),
                ("jamaica_page", ["jamaica", "jamaican"]),
                ("august_6", ["august 6", "6 august"]),
                ("year_1962", ["1962"]),
            ]
        )
        if source_id == "localnewsqa_explicit_0003201":
            checks.append(("emancipation_relation", ["emancipation day"]))
            checks.append(("week_long_celebration", ["week long cultural celebration"]))
        if "flag" in qn:
            checks.append(("flag_celebration", ["flag"]))
        if "parade" in qn:
            checks.append(("parade_celebration", ["parades", "parade"]))
        if "united kingdom" in qn:
            checks.append(("uk_relation", ["united kingdom"]))
    elif answer == "Labour Day":
        checks.extend(
            [
                ("answer_name", ["labour day", "labor day"]),
                ("jamaica_page", ["jamaica", "jamaican"]),
                ("holiday_status", ["national holiday", "public holiday"]),
                ("may_23", ["may 23", "23 may", "23rd of may"]),
            ]
        )
        if contains_any(qn, ["community", "clean up", "projects", "civic", "voluntary"]):
            checks.append(("community_projects", ["community projects", "improve public areas"]))
        if "schools" in qn:
            checks.append(("schools", ["schools"]))
    elif answer == "New Year's Day":
        checks.extend(
            [
                ("answer_name", ["new year's day", "new years day"]),
                ("jamaica_page", ["jamaica"]),
                ("holiday_status", ["public holiday"]),
                ("january_1", ["january 1", "jan 01", "01 jan"]),
            ]
        )
    elif answer == "Grade 6":
        checks.extend(
            [
                ("answer_grade", ["grade 6", "grade six"]),
                ("jamaica_page", ["jamaica"]),
                ("gsat", ["gsat", "grade six achievement test"]),
                ("primary_level", ["primary level", "primary schools"]),
            ]
        )
    elif answer == "Preparatory schools":
        checks.extend(
            [
                ("answer_name", ["preparatory schools"]),
                ("jamaica_page", ["jamaica"]),
                ("primary_level", ["primary level", "primary education"]),
                ("grades_1_to_6", ["grades 1 6", "grades 1-6"]),
            ]
        )
    elif answer == "Grade 1":
        checks.extend(
            [
                ("answer_grade", ["grade 1", "grade one"]),
                ("jamaica_page", ["jamaica"]),
                ("grade_one_profile", ["grade one individual learning profile"]),
                ("entering_grade_one", ["entering grade one", "child entering grade one"]),
            ]
        )
    else:
        raise ValueError(f"No relation requirements for {source_id} answer={answer!r}")
    return checks


def audit_relation_row(row: dict[str, Any], fetches: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fetch = fetches.get(row.get("target_evidence_url", ""), {})
    blob_norm = norm(evidence_blob(row, fetches))
    checks = relation_requirements(row)
    missing = [name for name, aliases in checks if not contains_any(blob_norm, aliases)]
    _, option_issues = option_shape_ok(row)
    if not fetch.get("ok"):
        missing.append("evidence_fetch_not_ok")
    if int(fetch.get("text_len", 0) or 0) < 500:
        missing.append("evidence_text_short")
    if not contains_any(norm(row.get("question", "")), ["jamaica", "jamaican"]):
        missing.append("question_lacks_country_marker")
    if row.get("target_evidence_title") in BROAD_TITLES:
        missing.append("remaining_broad_evidence_title")
    missing.extend(option_issues)
    return {
        "source_row_id": row.get("source_row_id", ""),
        "source_split_type": row.get("source_split_type", ""),
        "country": row.get("country", ""),
        "question": row.get("question", ""),
        "target_answer": row.get("target_answer", ""),
        "target_evidence_title": row.get("target_evidence_title", ""),
        "target_evidence_url": row.get("target_evidence_url", ""),
        "fetch_ok": bool(fetch.get("ok")),
        "fetch_text_len": fetch.get("text_len", 0),
        "required_checks": [name for name, _ in checks],
        "missing_checks": missing,
        "status": "pass" if not missing else "fail",
    }


def main() -> None:
    fetch_mod = load_module(FETCH_SCRIPT, "fetch32")
    broad_ids = read_csv_ids(BROAD_45_CSV)
    rows = read_jsonl(SPLIT_JSONL)
    repaired_rows, repair_log = apply_repairs(rows, broad_ids)
    needed_urls = {
        row["target_evidence_url"]
        for row in repaired_rows
        if row.get("source_row_id") in broad_ids and row.get("target_evidence_url")
    }
    fetches = load_fetch_cache(FETCH_CACHE)
    fetch_needed(needed_urls, fetches, fetch_mod)
    write_fetch_cache(FETCH_CACHE, fetches)

    final_45 = [row for row in repaired_rows if row.get("source_row_id") in broad_ids]
    relation_rows = [audit_relation_row(row, fetches) for row in final_45]
    failures = [row for row in relation_rows if row["status"] != "pass"]
    duplicate_source_ids = sum(
        1 for count in Counter(row.get("source_row_id", "") for row in repaired_rows).values() if count > 1
    )
    duplicate_questions = sum(
        1 for count in Counter(row.get("question", "") for row in repaired_rows).values() if count > 1
    )
    broad_rows_remaining = sum(1 for row in repaired_rows if row.get("target_evidence_title") in BROAD_TITLES)
    summary = {
        "input_jsonl": str(SPLIT_JSONL),
        "output_jsonl": str(SPLIT_JSONL),
        "output_csv": str(SPLIT_CSV),
        "rows_total": len(repaired_rows),
        "deep_audited_rows": len(final_45),
        "question_rewrites": sum(1 for row in repair_log if row["question_rewrite_applied"]),
        "question_fields_changed_this_run": sum(1 for row in repair_log if row["question_changed"]),
        "evidence_replacements": len(repair_log),
        "relation_failures": len(failures),
        "broad_country_specific_proxy_rows_remaining": broad_rows_remaining,
        "duplicate_source_ids": duplicate_source_ids,
        "duplicate_questions": duplicate_questions,
        "fetch_ok_for_deep_audited_unique_urls": sum(1 for url in needed_urls if fetches.get(url, {}).get("ok")),
        "deep_audited_unique_urls": len(needed_urls),
        "by_answer": dict(Counter(row.get("target_answer", "") for row in final_45)),
        "outputs": {
            "repair_log": str(OUT_DIR / "repair_log.csv"),
            "relation_audit": str(OUT_DIR / "relation_audit.csv"),
            "relation_failures": str(OUT_DIR / "relation_failures.csv"),
            "summary": str(OUT_DIR / "summary.json"),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(SPLIT_JSONL, repaired_rows)
    write_csv(SPLIT_CSV, repaired_rows)
    write_csv(OUT_DIR / "repair_log.csv", repair_log)
    write_csv(OUT_DIR / "relation_audit.csv", relation_rows)
    write_csv(OUT_DIR / "relation_failures.csv", failures)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
