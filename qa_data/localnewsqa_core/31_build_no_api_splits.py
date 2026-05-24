#!/usr/bin/env python3

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
QUALITY_AUDIT_DIR = ROOT / "qa_data/localnewsqa_core/runs/quality_audit"
SEMANTIC_SCORES = QUALITY_AUDIT_DIR / "semantic_quality_full_35874_scores.csv"
WEAK_LOCALE_FLAGS = QUALITY_AUDIT_DIR / "weak_locale_ambiguous_flags.csv"
LLM_ACCEPT_IDS = QUALITY_AUDIT_DIR / "llm_core_verification/llm_accept_core_ids.txt"
LLM_RESTORED_IDS = QUALITY_AUDIT_DIR / "llm_ambiguous_reject_adjudication/ambiguous_restorable_ids.txt"
DEFAULT_OUTDIR = QUALITY_AUDIT_DIR / "no_api_splits_20260515"

ALLOWED_EVIDENCE_ISSUES = {"evidence_display", "evidence_support"}
BAD_EXPLICIT_ISSUES = {"known_bad", "certification", "structure"}
COUNTRY_ORDER = [
    "United States",
    "Canada",
    "Jamaica",
    "India",
    "Pakistan",
    "Bangladesh",
    "Sri Lanka",
    "Hong Kong",
    "Malaysia",
    "Philippines",
    "Nigeria",
    "South Africa",
    "Kenya",
    "Ghana",
    "Tanzania",
    "United Kingdom",
    "Ireland",
]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_id_file(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def split_pipe(raw: str) -> set[str]:
    return {part.strip() for part in str(raw or "").split("|") if part.strip()}


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def option_parts(raw: str | list) -> list[str]:
    if isinstance(raw, list):
        return [str(part).strip() for part in raw if str(part).strip()]
    out = []
    for part in str(raw or "").split("||"):
        part = part.strip()
        if not part:
            continue
        out.append(re.sub(r"^[A-Da-d]\s*:\s*", "", part).strip())
    return out


def numeric(raw: str, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def target_answer(row: dict) -> str:
    return str(row.get("target_answer") or row.get("correct_answer") or "").strip()


def explicit_original_ok(row: dict) -> bool:
    issue_families = split_pipe(row.get("issue_families"))
    return (
        row.get("split") == "explicit"
        and row.get("target_source_certified") == "yes"
        and row.get("manual_review_status") != "manual_reject"
        and row.get("has_lower_bound_semantic_issue") != "yes"
        and row.get("semantic_risk_band") in {"clean", "low"}
        and not (issue_families & BAD_EXPLICIT_ISSUES)
    )


def salvage_target_ok(row: dict) -> bool:
    issue_families = split_pipe(row.get("issue_families"))
    return (
        row.get("split") == "ambiguous"
        and row.get("target_source_certified") == "yes"
        and row.get("manual_review_status") != "manual_reject"
        and row.get("has_lower_bound_semantic_issue") != "yes"
        and row.get("semantic_risk_band") == "clean"
        and not (issue_families - ALLOWED_EVIDENCE_ISSUES)
    )


def ambiguous_verifiable_ok(row: dict, weak_by_id: dict[str, dict]) -> bool:
    issue_families = split_pipe(row.get("issue_families"))
    weak = weak_by_id.get(row["id"], {})
    return (
        row.get("split") == "ambiguous"
        and row.get("target_source_certified") == "yes"
        and row.get("contrast_source_certified") == "yes"
        and row.get("manual_review_status") != "manual_reject"
        and row.get("has_lower_bound_semantic_issue") != "yes"
        and row.get("semantic_risk_band") == "clean"
        and weak.get("weak_locale_risk") in {"clean", "low"}
        and not (issue_families - ALLOWED_EVIDENCE_ISSUES)
    )


def question_with_country(country: str, question: str) -> str:
    question = str(question or "").strip()
    display_country = {
        "United States": "the United States",
        "United Kingdom": "the United Kingdom",
        "Philippines": "the Philippines",
    }.get(country, country)
    if not question:
        return f"In {display_country},"
    lowered = question.lower()
    country_lower = country.lower()
    if country_lower in lowered:
        return question
    first = question[0].lower() + question[1:] if len(question) > 1 else question.lower()
    return f"In {display_country}, {first}"


def evidence_excerpt(row: dict, side: str) -> str:
    return str(row.get(f"{side}_evidence_excerpt") or row.get(f"{side}_evidence_snippet") or "").strip()


def base_output_row(row: dict) -> dict:
    options = option_parts(row.get("options"))
    correct = target_answer(row)
    distractors = [opt for opt in options if normalize_text(opt) != normalize_text(correct)]
    return {
        "id": row["id"],
        "source_row_id": row["id"],
        "country": row.get("country", ""),
        "continent": row.get("continent", ""),
        "target_country": row.get("country", ""),
        "contrast_country": row.get("contrast_country", ""),
        "topic": row.get("topic", ""),
        "year": row.get("year", ""),
        "question": row.get("question", ""),
        "original_question": row.get("question", ""),
        "options": options,
        "correct_answer": correct,
        "distractors": distractors,
        "target_answer": correct,
        "contrast_answer": row.get("contrast_answer", ""),
        "evidence_hint": row.get("evidence_hint", ""),
        "target_evidence_url": row.get("target_evidence_url", ""),
        "target_evidence_title": row.get("target_evidence_title", ""),
        "target_evidence_excerpt": evidence_excerpt(row, "target"),
        "contrast_evidence_url": row.get("contrast_evidence_url", ""),
        "contrast_evidence_title": row.get("contrast_evidence_title", ""),
        "contrast_evidence_excerpt": evidence_excerpt(row, "contrast"),
        "semantic_risk_band": row.get("semantic_risk_band", ""),
        "semantic_error_probability": row.get("semantic_error_probability", ""),
        "evidence_support_score": row.get("evidence_support_score", ""),
        "issue_families": row.get("issue_families", ""),
    }


def make_targetqa_row(row: dict, source_split_type: str) -> dict:
    out = base_output_row(row)
    out.update(
        {
            "split_name": "LocalNewsQA-TargetQA",
            "split_type": "explicit",
            "split_family": "LocalNewsQA-Core",
            "ambiguity_flag": False,
            "source_split_type": source_split_type,
            "localized_question": question_with_country(row.get("country", ""), row.get("question", "")),
        }
    )
    if source_split_type == "ambiguous_salvaged_target":
        out["question"] = out["localized_question"]
    else:
        out["localized_question"] = out["question"]
    out["distractors"] = [
        opt for opt in out["options"] if normalize_text(opt) != normalize_text(out["correct_answer"])
    ]
    return out


def make_ambiguous_row(row: dict, weak_by_id: dict[str, dict], llm_ids: set[str], ordinal: int) -> dict:
    out = base_output_row(row)
    weak = weak_by_id.get(row["id"], {})
    out.update(
        {
            "id": f"localnewsqa_ambig_verifiable_{ordinal:04d}",
            "source_row_id": row["id"],
            "split_name": "LocalNewsQA-Ambiguous-Verifiable-1700",
            "split_type": "ambiguous",
            "split_family": "LocalNewsQA-Core",
            "ambiguity_flag": True,
            "source_split_type": "ambiguous",
            "weak_locale_risk": weak.get("weak_locale_risk", ""),
            "weak_locale_score": weak.get("weak_locale_score", ""),
            "weak_locale_flags": weak.get("weak_locale_flags", ""),
            "llm_accept_or_restored": row["id"] in llm_ids,
        }
    )
    return out


def country_sort_key(country: str) -> int:
    try:
        return COUNTRY_ORDER.index(country)
    except ValueError:
        return len(COUNTRY_ORDER)


def ambiguous_selection_key(row: dict, weak_by_id: dict[str, dict], llm_ids: set[str]) -> tuple:
    weak = weak_by_id.get(row["id"], {})
    return (
        0 if row["id"] in llm_ids else 1,
        0 if weak.get("weak_locale_risk") == "clean" else 1,
        -numeric(row.get("evidence_support_score"), -1.0),
        numeric(row.get("semantic_error_probability"), 1.0),
        row["id"],
    )


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def csv_value(value) -> str:
    if isinstance(value, list):
        return " || ".join(str(item) for item in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value or "")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def answer_count(options: list[str], answer: str) -> int:
    norm = normalize_text(answer)
    return sum(1 for opt in options if normalize_text(opt) == norm)


def validate_rows(targetqa: list[dict], ambiguous: list[dict], pool: list[dict]) -> dict:
    errors = []
    for name, rows in [
        ("targetqa", targetqa),
        ("ambiguous_1700", ambiguous),
        ("ambiguous_pool", pool),
    ]:
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)):
            errors.append(f"{name}: duplicate output ids")
        source_ids = [row["source_row_id"] for row in rows]
        if len(source_ids) != len(set(source_ids)):
            errors.append(f"{name}: duplicate source ids")
        for row in rows:
            options = row.get("options") or []
            if len(options) != 4:
                errors.append(f"{name}:{row['id']}: expected 4 options, got {len(options)}")
            if answer_count(options, row.get("correct_answer", "")) != 1:
                errors.append(f"{name}:{row['id']}: target answer not exactly once in options")
            if row.get("split_type") == "ambiguous" and answer_count(options, row.get("contrast_answer", "")) < 1:
                errors.append(f"{name}:{row['id']}: contrast answer absent from options")
    ambiguous_counts = Counter(row["country"] for row in ambiguous)
    for country in COUNTRY_ORDER:
        if ambiguous_counts[country] != 100:
            errors.append(f"ambiguous_1700:{country}: expected 100, got {ambiguous_counts[country]}")
    overlap = {row["source_row_id"] for row in targetqa} & {row["source_row_id"] for row in ambiguous}
    if overlap:
        errors.append(f"targetqa/ambiguous_1700 overlap: {len(overlap)} source ids")
    return {
        "ok": not errors,
        "error_count": len(errors),
        "errors": errors[:50],
    }


def summarize(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "by_country": dict(Counter(row.get("country", "") for row in rows)),
        "by_topic": dict(Counter(row.get("topic", "") for row in rows)),
        "by_source_split_type": dict(Counter(row.get("source_split_type", "") for row in rows)),
    }


def build_splits(outdir: Path) -> dict:
    source_rows = read_csv(SEMANTIC_SCORES)
    weak_by_id = {row["id"]: row for row in read_csv(WEAK_LOCALE_FLAGS)}
    llm_ids = load_id_file(LLM_ACCEPT_IDS) | load_id_file(LLM_RESTORED_IDS)

    ambiguous_candidates = [
        row for row in source_rows if ambiguous_verifiable_ok(row, weak_by_id)
    ]
    by_country: dict[str, list[dict]] = defaultdict(list)
    for row in ambiguous_candidates:
        by_country[row["country"]].append(row)

    selected_ambiguous_source_rows = []
    deficits = {}
    for country in COUNTRY_ORDER:
        rows = sorted(by_country.get(country, []), key=lambda r: ambiguous_selection_key(r, weak_by_id, llm_ids))
        if len(rows) < 100:
            deficits[country] = 100 - len(rows)
        selected_ambiguous_source_rows.extend(rows[:100])

    selected_source_ids = {row["id"] for row in selected_ambiguous_source_rows}

    targetqa_source_rows = []
    for row in source_rows:
        if explicit_original_ok(row):
            targetqa_source_rows.append((row, "explicit"))
        elif salvage_target_ok(row) and row["id"] not in selected_source_ids:
            targetqa_source_rows.append((row, "ambiguous_salvaged_target"))

    targetqa_rows = [
        make_targetqa_row(row, source_split_type)
        for row, source_split_type in sorted(
            targetqa_source_rows,
            key=lambda item: (
                country_sort_key(item[0].get("country", "")),
                0 if item[1] == "explicit" else 1,
                item[0]["id"],
            ),
        )
    ]
    ambiguous_pool_rows = [
        make_ambiguous_row(row, weak_by_id, llm_ids, idx)
        for idx, row in enumerate(
            sorted(
                ambiguous_candidates,
                key=lambda r: (
                    country_sort_key(r.get("country", "")),
                    ambiguous_selection_key(r, weak_by_id, llm_ids),
                ),
            ),
            start=1,
        )
    ]
    ambiguous_rows = [
        make_ambiguous_row(row, weak_by_id, llm_ids, idx)
        for idx, row in enumerate(
            sorted(
                selected_ambiguous_source_rows,
                key=lambda r: (
                    country_sort_key(r.get("country", "")),
                    ambiguous_selection_key(r, weak_by_id, llm_ids),
                ),
            ),
            start=1,
        )
    ]

    validation = validate_rows(targetqa_rows, ambiguous_rows, ambiguous_pool_rows)
    if not validation["ok"]:
        raise RuntimeError(json.dumps(validation, indent=2))

    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "targetqa_jsonl": outdir / "localnewsqa_targetqa_explicit_style.jsonl",
        "targetqa_csv": outdir / "localnewsqa_targetqa_explicit_style.csv",
        "ambiguous_jsonl": outdir / "localnewsqa_ambiguous_verifiable_1700.jsonl",
        "ambiguous_csv": outdir / "localnewsqa_ambiguous_verifiable_1700.csv",
        "ambiguous_pool_jsonl": outdir / "localnewsqa_ambiguous_verifiable_pool_4625.jsonl",
        "ambiguous_pool_csv": outdir / "localnewsqa_ambiguous_verifiable_pool_4625.csv",
        "summary": outdir / "summary.json",
    }
    write_jsonl(paths["targetqa_jsonl"], targetqa_rows)
    write_csv(paths["targetqa_csv"], targetqa_rows)
    write_jsonl(paths["ambiguous_jsonl"], ambiguous_rows)
    write_csv(paths["ambiguous_csv"], ambiguous_rows)
    write_jsonl(paths["ambiguous_pool_jsonl"], ambiguous_pool_rows)
    write_csv(paths["ambiguous_pool_csv"], ambiguous_pool_rows)

    summary = {
        "filters": {
            "targetqa_original_explicit": "explicit; target_source_certified=yes; not manual_reject; no lower-bound semantic issue; semantic risk clean/low; no known_bad/certification/structure issue family",
            "targetqa_salvaged_ambiguous": "ambiguous; target_source_certified=yes; not manual_reject; no lower-bound semantic issue; semantic risk clean; issue families limited to evidence_display/evidence_support; excludes selected ambiguous 1700 source ids",
            "ambiguous_verifiable": "ambiguous; target and contrast certified; not manual_reject; no lower-bound semantic issue; semantic risk clean; weak-locale risk clean/low; issue families limited to evidence_display/evidence_support",
        },
        "targetqa": summarize(targetqa_rows),
        "ambiguous_1700": {
            **summarize(ambiguous_rows),
            "weak_locale_risk": dict(Counter(row.get("weak_locale_risk", "") for row in ambiguous_rows)),
            "llm_accept_or_restored": dict(Counter(str(row.get("llm_accept_or_restored")) for row in ambiguous_rows)),
        },
        "ambiguous_pool": {
            **summarize(ambiguous_pool_rows),
            "weak_locale_risk": dict(Counter(row.get("weak_locale_risk", "") for row in ambiguous_pool_rows)),
            "llm_accept_or_restored": dict(Counter(str(row.get("llm_accept_or_restored")) for row in ambiguous_pool_rows)),
        },
        "ambiguous_deficits": deficits,
        "validation": validation,
        "paths": {key: str(value) for key, value in paths.items()},
    }
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build no-API LocalNewsQA TargetQA and verifiable ambiguous splits.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_splits(args.outdir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
