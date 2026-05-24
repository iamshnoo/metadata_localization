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
BASE = ROOT / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515"
DEFAULT_OUTDIR = BASE / "web_semantic_gold_ambiguous_1700"
DEFAULT_POOL = BASE / "localnewsqa_ambiguous_verifiable_pool_4625.jsonl"
DEFAULT_FETCH_CACHE = BASE / "web_audit_ambiguous_1700/url_fetches.jsonl"

AUDIT_SCRIPT = ROOT / "qa_data/localnewsqa_core/32_web_audit_ambiguous_verifiable.py"
GOLD_SCRIPT = ROOT / "qa_data/localnewsqa_core/41_build_semantic_gold_ambiguous.py"
REPAIR_SCRIPT = ROOT / "qa_data/localnewsqa_core/43_repair_semantic_gold_questions.py"
RELATION_SCRIPT = ROOT / "qa_data/localnewsqa_core/44_relation_support_audit.py"
WEAK_SCRIPT = ROOT / "qa_data/localnewsqa_core/26_flag_weak_locale_ambiguous.py"
REVIEWER_SCRIPT = ROOT / "qa_data/localnewsqa_core/47_reviewer_risk_audit.py"


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


def relation_pass(row: dict, fetches: dict[str, dict], relation_mod: Any, audit_mod: Any) -> tuple[bool, list[dict]]:
    audits = [
        relation_mod.side_relation_audit(row, "target", fetches, audit_mod),
        relation_mod.side_relation_audit(row, "contrast", fetches, audit_mod),
    ]
    return all(item["severity"] == "pass" for item in audits), audits


def structural_pass(row: dict, audit_mod: Any, repair_mod: Any) -> tuple[bool, list[str]]:
    reasons = []
    options = audit_mod.option_parts(row.get("options", []))
    norm_options = [audit_mod.normalize_text(option) for option in options]
    if len(options) != 4:
        reasons.append("option_count_not_4")
    if len(norm_options) != len(set(norm_options)):
        reasons.append("duplicate_normalized_options")
    if audit_mod.answer_count(options, row.get("target_answer", "")) != 1:
        reasons.append("target_answer_not_exactly_once")
    if audit_mod.answer_count(options, row.get("contrast_answer", "")) != 1:
        reasons.append("contrast_answer_not_exactly_once")
    if audit_mod.normalize_text(row.get("target_answer")) == audit_mod.normalize_text(row.get("contrast_answer")):
        reasons.append("same_target_contrast_answer")
    if row.get("country") == row.get("contrast_country"):
        reasons.append("same_target_contrast_country")
    if row.get("target_evidence_url") == row.get("contrast_evidence_url"):
        reasons.append("same_target_contrast_evidence_url")
    if row.get("weak_locale_risk") != "clean" or row.get("weak_locale_flags"):
        reasons.append("weak_locale_not_clean")
    if repair_mod.question_quality_issues(row.get("question", "")):
        reasons.append("question_quality_artifact")
    return not reasons, reasons


def selected_fetch_rows(rows: list[dict], fetches: dict[str, dict]) -> list[dict]:
    urls = sorted(
        {
            url
            for row in rows
            for url in [row.get("target_evidence_url", ""), row.get("contrast_evidence_url", "")]
            if url
        }
    )
    return [fetches[url] for url in urls if url in fetches]


def refresh_short_selected_fetches(rows: list[dict], fetches: dict[str, dict], audit_mod: Any) -> list[dict]:
    refreshed = []
    for row in rows:
        audit = audit_mod.audit_row(row, fetches)
        warning_parts = {part.strip() for part in audit["warnings"].split(" | ") if part.strip()}
        for side in ["target", "contrast"]:
            warning = f"{side}_evidence_text_short"
            url = row.get(f"{side}_evidence_url", "")
            old_fetch = fetches.get(url, {})
            if warning not in warning_parts or not url or not old_fetch.get("ok"):
                continue
            new_fetch = audit_mod.fetch_url(url, timeout=20, delay=0)
            if new_fetch.get("ok") and int(new_fetch.get("text_len", 0) or 0) > int(old_fetch.get("text_len", 0) or 0):
                fetches[url] = new_fetch
                refreshed.append(
                    {
                        "source_row_id": row.get("source_row_id", ""),
                        "side": side,
                        "url": url,
                        "old_text_len": old_fetch.get("text_len", 0),
                        "new_text_len": new_fetch.get("text_len", 0),
                        "status": "refreshed_full_html",
                    }
                )
            else:
                refreshed.append(
                    {
                        "source_row_id": row.get("source_row_id", ""),
                        "side": side,
                        "url": url,
                        "old_text_len": old_fetch.get("text_len", 0),
                        "new_text_len": new_fetch.get("text_len", 0),
                        "status": "kept_existing_fetch",
                        "error": new_fetch.get("error", ""),
                    }
                )
    return refreshed


def cue_norm(value: Any) -> str:
    text = str(value or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def high_confidence_question_cues(row: dict) -> list[dict]:
    """Return reviewer-visible quoted titles/acronyms that evidence should support literally."""
    question = str(row.get("question", ""))
    answer_norms = {
        cue_norm(row.get("target_answer", "")),
        cue_norm(row.get("contrast_answer", "")),
    }
    cues: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, cue: str) -> None:
        cleaned = cue.strip(" \t\r\n\"'.,;:!?()[]{}")
        norm = cue_norm(cleaned)
        if len(norm) < 3 or norm in answer_norms:
            return
        if any(norm and norm == answer for answer in answer_norms):
            return
        key = (kind, norm)
        if key not in seen:
            seen.add(key)
            cues.append({"kind": kind, "cue": cleaned, "norm": norm})

    for match in re.finditer(r'["“]([^"“”]{3,80})["”]', question):
        add("quoted", match.group(1))
    for match in re.finditer(r"(?<![A-Za-z0-9])['‘]([^'‘’]{3,80})['’](?![A-Za-z0-9])", question):
        add("quoted", match.group(1))

    acronym_pattern = re.compile(r"\b[A-Z][A-Z0-9&./-]{1,}[A-Z0-9]\b")
    for match in acronym_pattern.finditer(question):
        token = match.group(0).strip(".-/")
        letters = re.sub(r"[^A-Z]", "", token)
        if len(letters) >= 3:
            add("acronym", token)

    return cues


def evidence_cue_text(row: dict, side: str, fetches: dict[str, dict]) -> str:
    url = row.get(f"{side}_evidence_url", "")
    fetch = fetches.get(url, {})
    parts = [
        row.get(f"{side}_evidence_title", ""),
        row.get(f"{side}_evidence_excerpt", ""),
        fetch.get("title", ""),
        fetch.get("text", ""),
        fetch.get("markdown", ""),
    ]
    return cue_norm(" ".join(str(part or "") for part in parts))


def high_confidence_cue_issues(row: dict, fetches: dict[str, dict]) -> list[dict]:
    issues = []
    target_text = evidence_cue_text(row, "target", fetches)
    contrast_text = evidence_cue_text(row, "contrast", fetches)
    for cue in high_confidence_question_cues(row):
        missing_sides = []
        if cue["norm"] not in target_text:
            missing_sides.append("target")
        if cue["norm"] not in contrast_text:
            missing_sides.append("contrast")
        if missing_sides:
            issues.append(
                {
                    "kind": cue["kind"],
                    "cue": cue["cue"],
                    "missing_sides": ",".join(missing_sides),
                }
            )
    return issues


def selection_key(row: dict) -> tuple:
    return (
        int(int(row.get("reviewer_visible_cue_risk_count", 0) or 0) > 0),
        int(row.get("reviewer_visible_cue_risk_count", 0) or 0),
        *gold_selection_key(row),
    )


def gold_selection_key(row: dict) -> tuple:
    return (
        int(row.get("semantic_gold_warning_count", 0) > 0),
        0 if row.get("weak_locale_risk") == "clean" else 1,
        0 if str(row.get("llm_accept_or_restored", "")).lower() == "true" else 1,
        -float(row.get("evidence_support_score", 0) or 0),
        float(row.get("semantic_error_probability", 1) or 1),
        row.get("source_row_id", ""),
    )


def diversity_norm(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def question_template_key(question: str) -> str:
    text = str(question or "").lower()
    text = re.sub(r"\b(?:19|20)\d{2}\b", "YEAR", text)
    text = re.sub(
        r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
        "MONTH",
        text,
    )
    text = re.sub(r"[^a-z0-9]+", " ", text)
    words = text.split()
    return " ".join(words[:8])


def select_diverse_country_rows(pool: list[dict], limit: int = 100) -> list[dict]:
    remaining = sorted(pool, key=selection_key)
    selected = []
    pair_counts: Counter[tuple[str, str]] = Counter()
    target_counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()

    while remaining and len(selected) < limit:
        def diversity_key(row: dict) -> tuple:
            pair = (diversity_norm(row.get("target_answer", "")), diversity_norm(row.get("contrast_answer", "")))
            target = diversity_norm(row.get("target_answer", ""))
            template = question_template_key(row.get("question", ""))
            topic = diversity_norm(row.get("topic", ""))
            return (
                int(int(row.get("reviewer_visible_cue_risk_count", 0) or 0) > 0),
                pair_counts[pair],
                target_counts[target],
                template_counts[template],
                topic_counts[topic] // 15,
                selection_key(row),
            )

        best_index, best_row = min(enumerate(remaining), key=lambda item: diversity_key(item[1]))
        selected.append(best_row)
        remaining.pop(best_index)
        pair = (diversity_norm(best_row.get("target_answer", "")), diversity_norm(best_row.get("contrast_answer", "")))
        target = diversity_norm(best_row.get("target_answer", ""))
        template = question_template_key(best_row.get("question", ""))
        topic = diversity_norm(best_row.get("topic", ""))
        pair_counts[pair] += 1
        target_counts[target] += 1
        template_counts[template] += 1
        topic_counts[topic] += 1

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build relation-strict semantic-gold ambiguous split.")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--fetch-cache", type=Path, default=DEFAULT_FETCH_CACHE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    audit_mod = load_module(AUDIT_SCRIPT, "audit32")
    gold_mod = load_module(GOLD_SCRIPT, "gold41")
    repair_mod = load_module(REPAIR_SCRIPT, "repair43")
    relation_mod = load_module(RELATION_SCRIPT, "relation44")
    weak_mod = load_module(WEAK_SCRIPT, "weak26")
    reviewer_mod = load_module(REVIEWER_SCRIPT, "reviewer47")

    rows = read_jsonl(args.pool)
    fetches = audit_mod.load_existing_fetches(args.fetch_cache)
    pages = gold_mod.fetch_titles(set(gold_mod.OVERRIDES.values()))
    url_pages = {url: gold_mod.fetch_direct_url(url) for url in sorted(set(gold_mod.URL_OVERRIDES.values()))}

    # Use full extracts for deterministic override pages when available; intro-only
    # extracts can miss relation/temporal details in officeholder rows.
    override_urls = {page["url"] for page in pages.values() if page.get("ok") and page.get("url")}
    full_override_fetches = gold_mod.fetch_full_wikipedia_pages(override_urls, sleep=0.05)
    for url, fetch in full_override_fetches.items():
        if fetch.get("ok"):
            fetches[url] = fetch

    candidates_by_country: dict[str, list[dict]] = defaultdict(list)
    rejected = []
    override_logs = []
    repair_logs = []
    relation_audits = []
    for index, row in enumerate(rows, start=1):
        repaired_evidence, logs = gold_mod.apply_overrides(row, pages, url_pages, fetches)
        override_logs.extend(logs)
        for side in ["target", "contrast"]:
            url = repaired_evidence.get(f"{side}_evidence_url", "")
            if url in full_override_fetches and full_override_fetches[url].get("ok"):
                fetches[url] = full_override_fetches[url]

        web_audit = audit_mod.audit_row(repaired_evidence, fetches)
        risks = gold_mod.side_risk(repaired_evidence, "target", fetches) + gold_mod.side_risk(
            repaired_evidence, "contrast", fetches
        )
        if not gold_mod.acceptable_audit(web_audit) or risks:
            rejected.append(
                {
                    "source_row_id": row["source_row_id"],
                    "country": row["country"],
                    "stage": "web_or_evidence_specificity",
                    "web_severity": web_audit["severity"],
                    "web_failures": web_audit["failures"],
                    "web_warnings": web_audit["warnings"],
                    "semantic_risks": " | ".join(risks),
                }
            )
            continue

        repaired_question, repair_log = repair_mod.repair_row(repaired_evidence, weak_mod)
        if repair_log:
            repair_logs.append(repair_log)
        web_audit = audit_mod.audit_row(repaired_question, fetches)
        warning_parts = [part.strip() for part in web_audit["warnings"].split(" | ") if part.strip()]
        repaired_question["semantic_gold_status"] = "pass"
        repaired_question["semantic_gold_notes"] = web_audit["warnings"]
        repaired_question["semantic_gold_mechanical_web_audit_severity"] = web_audit["severity"]
        repaired_question["semantic_gold_warning_count"] = len(warning_parts)
        repaired_question["web_audit_severity"] = "pass" if gold_mod.acceptable_audit(web_audit) else web_audit["severity"]
        repaired_question["web_audit_failures"] = "" if gold_mod.acceptable_audit(web_audit) else web_audit["failures"]
        repaired_question["web_audit_warnings"] = web_audit["warnings"]

        ok_structure, structure_reasons = structural_pass(repaired_question, audit_mod, repair_mod)
        ok_relation, side_audits = relation_pass(repaired_question, fetches, relation_mod, audit_mod)
        relation_audits.extend(side_audits)
        if not ok_structure or not ok_relation:
            rejected.append(
                {
                    "source_row_id": row["source_row_id"],
                    "country": row["country"],
                    "stage": "structure_or_relation",
                    "structure_reasons": " | ".join(structure_reasons),
                    "relation_severities": " | ".join(f"{item['side']}:{item['severity']}" for item in side_audits),
                    "relation_issues": " || ".join(
                        f"{item['side']}:{item['issues']}" for item in side_audits if item["issues"]
                    ),
                    "question": repaired_question["question"],
                    "target_answer": repaired_question["target_answer"],
                    "contrast_answer": repaired_question["contrast_answer"],
                }
            )
            continue

        repaired_question["relation_support_status"] = "pass"
        repaired_question["relation_target_strong"] = side_audits[0]["strong_relation"]
        repaired_question["relation_contrast_strong"] = side_audits[1]["strong_relation"]
        repaired_question["relation_target_cue_hits"] = side_audits[0]["cue_hits"]
        repaired_question["relation_contrast_cue_hits"] = side_audits[1]["cue_hits"]
        reviewer_risks = reviewer_mod.reviewer_risks(repaired_question, fetches, audit_mod, weak_mod)
        repaired_question["reviewer_visible_cue_risk_count"] = len(reviewer_risks)
        repaired_question["reviewer_visible_cue_risks"] = " | ".join(
            f"{risk['risk']}:{risk['cue']}" for risk in reviewer_risks
        )

        high_conf_issues = high_confidence_cue_issues(repaired_question, fetches)
        if high_conf_issues:
            rejected.append(
                {
                    "source_row_id": row["source_row_id"],
                    "country": row["country"],
                    "stage": "high_confidence_question_cue",
                    "question": repaired_question["question"],
                    "target_answer": repaired_question["target_answer"],
                    "contrast_answer": repaired_question["contrast_answer"],
                    "high_confidence_cue_issues": " | ".join(
                        f"{issue['kind']}:{issue['cue']} missing={issue['missing_sides']}"
                        for issue in high_conf_issues
                    ),
                }
            )
            continue

        candidates_by_country[repaired_question["country"]].append(repaired_question)
        if index % 500 == 0:
            print(f"processed {index}/{len(rows)}", flush=True)

    selected = []
    deficits = {}
    for country in gold_mod.COUNTRY_ORDER:
        pool = sorted(candidates_by_country.get(country, []), key=selection_key)
        if len(pool) < 100:
            deficits[country] = 100 - len(pool)
        selected.extend(select_diverse_country_rows(pool, limit=100))

    selected.sort(key=lambda row: (gold_mod.COUNTRY_ORDER.index(row["country"]), selection_key(row)))
    final_rows = []
    for idx, row in enumerate(selected, start=1):
        out = dict(row)
        out["id"] = f"localnewsqa_ambig_semantic_gold_{idx:04d}"
        out["split_name"] = "LocalNewsQA-Ambiguous-Semantic-Gold-1700"
        final_rows.append(out)

    validation_errors = gold_mod.validate(final_rows)
    if deficits:
        validation_errors.append(f"relation-strict candidate deficits: {deficits}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.outdir / "localnewsqa_ambiguous_semantic_gold_1700.jsonl"
    csv_path = args.outdir / "localnewsqa_ambiguous_semantic_gold_1700.csv"
    fetch_path = args.outdir / "semantic_gold_selected_evidence_fetches.jsonl"
    summary_path = args.outdir / "relation_strict_build_summary.json"
    rejected_path = args.outdir / "relation_strict_rejected_pool_rows.csv"
    relation_path = args.outdir / "relation_strict_candidate_side_audit.csv"
    repair_log_path = args.outdir / "semantic_gold_question_repair_log.csv"
    override_log_path = args.outdir / "semantic_gold_override_log.csv"
    short_refresh_path = args.outdir / "semantic_gold_selected_short_evidence_refresh_log.csv"

    short_refresh_logs = refresh_short_selected_fetches(final_rows, fetches, audit_mod)
    for row in final_rows:
        refreshed_risks = reviewer_mod.reviewer_risks(row, fetches, audit_mod, weak_mod)
        row["reviewer_visible_cue_risk_count"] = len(refreshed_risks)
        row["reviewer_visible_cue_risks"] = " | ".join(
            f"{risk['risk']}:{risk['cue']}" for risk in refreshed_risks
        )
        refreshed_high_conf = high_confidence_cue_issues(row, fetches)
        row["high_confidence_question_cue_risk_count"] = len(refreshed_high_conf)
        row["high_confidence_question_cue_risks"] = " | ".join(
            f"{issue['kind']}:{issue['cue']} missing={issue['missing_sides']}" for issue in refreshed_high_conf
        )
    write_jsonl(jsonl_path, final_rows)
    gold_mod.write_csv(csv_path, final_rows)
    write_jsonl(fetch_path, selected_fetch_rows(final_rows, fetches))
    write_csv(rejected_path, rejected)
    write_csv(relation_path, relation_audits)
    write_csv(repair_log_path, repair_logs)
    write_csv(override_log_path, override_logs)
    write_csv(short_refresh_path, short_refresh_logs)

    summary = {
        "pool": str(args.pool),
        "rows": len(final_rows),
        "valid": not validation_errors,
        "validation_errors": validation_errors,
        "candidate_counts_by_country": {
            country: len(candidates_by_country.get(country, [])) for country in gold_mod.COUNTRY_ORDER
        },
        "country_counts": dict(Counter(row["country"] for row in final_rows)),
        "rejected_stage_counts": dict(Counter(row["stage"] for row in rejected)),
        "high_confidence_question_cue_rejected_rows": sum(
            1 for row in rejected if row.get("stage") == "high_confidence_question_cue"
        ),
        "relation_candidate_side_severity_counts": dict(Counter(row["severity"] for row in relation_audits)),
        "selected_relation_strong_side_counts": {
            "target": sum(1 for row in final_rows if row.get("relation_target_strong") is True),
            "contrast": sum(1 for row in final_rows if row.get("relation_contrast_strong") is True),
        },
        "question_repaired_rows": len(repair_logs),
        "override_counts": dict(Counter(row["status"] for row in override_logs)),
        "selected_short_evidence_refresh_counts": dict(Counter(row["status"] for row in short_refresh_logs)),
        "reviewer_visible_cue_risk_candidates_by_country": {
            country: sum(
                1
                for row in candidates_by_country.get(country, [])
                if int(row.get("reviewer_visible_cue_risk_count", 0) or 0) > 0
            )
            for country in gold_mod.COUNTRY_ORDER
        },
        "selected_reviewer_visible_cue_risk_row_count": sum(
            1 for row in final_rows if int(row.get("reviewer_visible_cue_risk_count", 0) or 0) > 0
        ),
        "selected_high_confidence_question_cue_risk_row_count": sum(
            1 for row in final_rows if int(row.get("high_confidence_question_cue_risk_count", 0) or 0) > 0
        ),
        "selected_max_answer_pair_count": max(
            Counter((row.get("target_answer", ""), row.get("contrast_answer", "")) for row in final_rows).values()
        )
        if final_rows
        else 0,
        "selected_answer_pairs_repeated_over_10": {
            f"{pair[0]} / {pair[1]}": count
            for pair, count in Counter(
                (row.get("target_answer", ""), row.get("contrast_answer", "")) for row in final_rows
            ).items()
            if count > 10
        },
        "paths": {
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
            "selected_evidence_fetches": str(fetch_path),
            "summary": str(summary_path),
            "rejected": str(rejected_path),
            "relation_candidate_side_audit": str(relation_path),
            "question_repair_log": str(repair_log_path),
            "override_log": str(override_log_path),
            "selected_short_evidence_refresh_log": str(short_refresh_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if validation_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
