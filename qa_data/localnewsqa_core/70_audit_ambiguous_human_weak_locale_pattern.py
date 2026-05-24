#!/usr/bin/env python3
"""No-API audit for the human weak-locale reject pattern in ambiguous rows.

This script does not change the dataset.  It creates an internal triage report
for rows that resemble the annotator rejects where the target and contrast
answers are factual but the question is only weakly / awkwardly locale-dependent.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "qa_data/localnewsqa_core/final_gold_20260516/"
    / "localnewsqa_gold_explicit17000_ambiguous1700.jsonl"
)
DEFAULT_FETCHES = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/"
    / "web_semantic_gold_ambiguous_1700/semantic_gold_selected_evidence_fetches.jsonl"
)
DEFAULT_OUTDIR = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/"
    / "web_semantic_gold_ambiguous_1700/human_weak_locale_pattern_audit_20260519"
)


VAGUE_NEWS_PATTERNS = [
    r"\bcoverage\b",
    r"\bstor(?:y|ies)\b",
    r"\bappearing in news\b",
    r"\blocal news\b",
    r"\bheadline(?:s)?\b",
    r"\bmajor .*? coverage\b",
    r"\bmajor .*? stor(?:y|ies)\b",
    r"\bcentral to\b",
    r"\bflagship\b",
    r"\bprominent(?:ly)?\b",
    r"\boften linked\b",
    r"\bmost often linked\b",
    r"\bwidely linked\b",
    r"\bassociated with\b",
    r"\bnews before\b",
]

TIME_FRAME_PATTERNS = [
    r"\bwhen\b",
    r"\bduring\b",
    r"\bbefore\b",
    r"\bafter\b",
    r"\bin 20\d\d\b",
    r"\bin 19\d\d\b",
]

ROLE_WORDS = [
    "president",
    "prime minister",
    "minister",
    "governor",
    "mayor",
    "secretary",
    "speaker",
    "chancellor",
]

ROLE_UNAVAILABLE_BY_COUNTRY = {
    "prime minister": {
        "United States",
        "Philippines",
        "Nigeria",
        "Ghana",
        "Tanzania",
        "South Africa",
        "Hong Kong",
    },
    "president": {
        "United Kingdom",
        "Canada",
        "Jamaica",
        "Malaysia",
        "Hong Kong",
    },
}

RESTRUCTURING_PATTERNS = [
    r"\bbefore\b.*\b(functions?|responsibilit(?:y|ies)|powers?)\b.*\b(moved|transferred|shifted)\b",
    r"\b(functions?|responsibilit(?:y|ies)|powers?)\b.*\b(moved|transferred|shifted)\b",
    r"\bmoved to another agency\b",
    r"\breplaced by\b",
    r"\babolished\b",
    r"\bmerged\b",
    r"\brestructured\b",
]

GENERIC_TITLE_TERMS = {
    "president",
    "prime minister",
    "minister",
    "government",
    "office",
    "authority",
    "commission",
    "department",
    "agency",
    "council",
    "court",
    "parliament",
    "senate",
    "house",
    "university",
    "city",
    "bank",
    "airport",
    "party",
}

QUESTION_STARTERS = {
    "A",
    "An",
    "After",
    "Before",
    "During",
    "For",
    "From",
    "How",
    "In",
    "On",
    "The",
    "What",
    "When",
    "Where",
    "Which",
    "Who",
}

COUNTRY_WORDS = {
    "United States",
    "United Kingdom",
    "South Africa",
    "Hong Kong",
    "Sri Lanka",
    "Canada",
    "Jamaica",
    "India",
    "Pakistan",
    "Bangladesh",
    "Malaysia",
    "Philippines",
    "Nigeria",
    "Kenya",
    "Ghana",
    "Tanzania",
    "Ireland",
}


def norm(text: Any) -> str:
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def has_pattern(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


def pattern_hits(text: str, patterns: list[str]) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text, flags=re.I)]


def phrase_in(text: str, phrase: str) -> bool:
    phrase_norm = norm(phrase)
    if not phrase_norm:
        return False
    return f" {phrase_norm} " in f" {norm(text)} "


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_fetches(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    fetches = {}
    for row in load_jsonl(path):
        url = row.get("url")
        if url:
            fetches[url] = row
    return fetches


def side_text(row: dict[str, Any], side: str, fetches: dict[str, dict[str, Any]]) -> str:
    url = row.get(f"{side}_evidence_url", "")
    fetch = fetches.get(url, {})
    parts = [
        row.get(f"{side}_evidence_title", ""),
        row.get(f"{side}_evidence_excerpt", ""),
        row.get(f"{side}_evidence_snippet", ""),
        fetch.get("title", ""),
        fetch.get("text", ""),
    ]
    return " ".join(str(part or "") for part in parts)


def proper_spans(text: str) -> list[str]:
    spans: list[str] = []
    for match in re.finditer(
        r"\b(?:[A-Z][A-Za-z&.'-]+|[A-Z]{2,})(?:\s+(?:of|the|and|for|in|on|to|[A-Z][A-Za-z&.'-]+|[A-Z]{2,})){0,8}",
        text or "",
    ):
        span = re.sub(r"\s+", " ", match.group(0)).strip(" ,.;!?()[]{}")
        words = span.split()
        if not span or span in QUESTION_STARTERS or span in COUNTRY_WORDS:
            continue
        if words and words[0] in QUESTION_STARTERS:
            continue
        if len(words) == 1 and (span in QUESTION_STARTERS or span.lower() in GENERIC_TITLE_TERMS):
            continue
        if len(norm(span)) < 3 or norm(span).isdigit():
            continue
        spans.append(span)
    return list(dict.fromkeys(spans))


def acronym_spans(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\b[A-Z]{2,}(?:[-/][A-Z]{2,})*\b", text or "")))


def title_is_weak_acronym(row: dict[str, Any], side: str) -> bool:
    title = str(row.get(f"{side}_evidence_title", "") or "").strip()
    answer = str(row.get(f"{side}_answer", "") or "").strip()
    if not title:
        return True
    title_norm = norm(title)
    if "disambiguation" in title_norm:
        return True
    if re.fullmatch(r"[A-Z0-9]{2,8}", title) and title == answer:
        return True
    return False


def role_terms_in_question(question_norm: str) -> list[str]:
    return [role for role in ROLE_WORDS if f" {role} " in f" {question_norm} "]


def explicit_role_mismatch(row: dict[str, Any], role_terms: list[str]) -> list[str]:
    flags = []
    for role in role_terms:
        if role in ROLE_UNAVAILABLE_BY_COUNTRY and row.get("contrast_country") in ROLE_UNAVAILABLE_BY_COUNTRY[role]:
            flags.append(f"question_role_unavailable_for_contrast_country:{role}")
    return flags


def answer_type(row: dict[str, Any], side: str) -> str:
    text = norm(
        " ".join(
            [
                row.get(f"{side}_answer", ""),
                row.get(f"{side}_evidence_title", ""),
                row.get("question", ""),
            ]
        )
    )
    if any(term in text for term in ["president", "prime minister", "minister", "governor", "mayor", "speaker"]):
        return "political_role_or_person"
    if any(term in text for term in ["commission", "authority", "agency", "office", "department", "regulator", "council", "bureau", "board"]):
        return "agency_or_office"
    if any(term in text for term in ["city", "county", "district", "province", "airport", "road", "bridge", "station"]):
        return "place_or_infrastructure"
    if any(term in text for term in ["airline", "bank", "company", "corporation", "exchange", "business"]):
        return "company_or_market"
    if any(term in text for term in ["university", "college", "school"]):
        return "education"
    if any(term in text for term in ["party", "team", "club", "league"]):
        return "organization"
    return "unknown"


def target_only_anchor_cues(row: dict[str, Any], target_text: str, contrast_text: str) -> list[str]:
    question = str(row.get("question", "") or "")
    hint = str(row.get("evidence_hint", "") or "")
    answer_blob = norm(" ".join([row.get("target_answer", ""), row.get("contrast_answer", "")]))
    candidates = []
    candidates.extend(proper_spans(question))
    candidates.extend(acronym_spans(question))
    candidates.extend(proper_spans(hint))
    candidates.extend(acronym_spans(hint))
    clean = []
    for cue in candidates:
        cue_norm = norm(cue)
        if not cue_norm or cue_norm in answer_blob:
            continue
        if cue_norm in {norm(c) for c in COUNTRY_WORDS}:
            continue
        if len(cue_norm) < 4:
            continue
        target_hit = phrase_in(target_text, cue)
        contrast_hit = phrase_in(contrast_text, cue)
        if target_hit and not contrast_hit:
            clean.append(cue)
    return list(dict.fromkeys(clean))


def audit_row(row: dict[str, Any], fetches: dict[str, dict[str, Any]]) -> dict[str, Any]:
    q = str(row.get("question", "") or "")
    q_norm = norm(q)
    target_text = side_text(row, "target", fetches)
    contrast_text = side_text(row, "contrast", fetches)
    flags: list[str] = []
    notes: list[str] = []
    score = 0

    vague_hits = pattern_hits(q, VAGUE_NEWS_PATTERNS)
    time_hits = pattern_hits(q, TIME_FRAME_PATTERNS)
    role_terms = role_terms_in_question(q_norm)
    role_mismatch_flags = explicit_role_mismatch(row, role_terms)
    restructure_hits = pattern_hits(q, RESTRUCTURING_PATTERNS)
    target_only_cues = target_only_anchor_cues(row, target_text, contrast_text)

    if vague_hits:
        flags.append("vague_news_or_coverage_framing")
        score += 1
        notes.append("vague=" + "|".join(vague_hits[:4]))
    if time_hits:
        flags.append("time_framed_question")
        score += 1
    if role_terms and time_hits:
        flags.append("office_role_time_analogy")
        score += 1
    if role_terms and vague_hits:
        flags.append("political_role_plus_vague_coverage")
        score += 3
    if role_mismatch_flags:
        flags.extend(role_mismatch_flags)
        score += 3
    if restructure_hits:
        flags.append("institution_restructuring_frame")
        score += 5
    if target_only_cues:
        flags.append("target_only_named_anchor_in_question_or_hint")
        score += min(3, 1 + len(target_only_cues))
        notes.append("target_only_cues=" + "|".join(target_only_cues[:5]))

    target_weak_title = title_is_weak_acronym(row, "target")
    contrast_weak_title = title_is_weak_acronym(row, "contrast")
    if target_weak_title or contrast_weak_title:
        flags.append("weak_or_acronym_evidence_title")
        score += 1

    target_type = answer_type(row, "target")
    contrast_type = answer_type(row, "contrast")
    if target_type != "unknown" and contrast_type != "unknown" and target_type != contrast_type:
        flags.append(f"rough_answer_type_mismatch:{target_type}_vs_{contrast_type}")
        score += 3

    if row.get("relation_target_strong") is False or row.get("relation_contrast_strong") is False:
        flags.append("existing_relation_probe_not_strong_on_one_side")
        score += 1

    # A lone generic "coverage" cue is common in this dataset and is not enough
    # to call the row close to the human reject pattern.  Demote it unless it
    # combines with role/time, target-only anchors, or restructuring.
    core_pattern = any(
        flag.startswith("office_role_time")
        or flag.startswith("political_role_plus")
        or flag.startswith("question_role_")
        or flag == "institution_restructuring_frame"
        or flag == "target_only_named_anchor_in_question_or_hint"
        or flag.startswith("rough_answer_type_mismatch")
        for flag in flags
    )
    if score == 0:
        band = "clean"
    elif not core_pattern and score < 4:
        band = "low_watch"
    elif score >= 6:
        band = "high"
    elif score >= 4:
        band = "medium"
    elif score > 0:
        band = "low_watch"
    else:
        band = "clean"

    return {
        "id": row.get("id", ""),
        "source_row_id": row.get("source_row_id", ""),
        "country": row.get("country", ""),
        "contrast_country": row.get("contrast_country", ""),
        "topic": row.get("topic", ""),
        "question": q,
        "target_answer": row.get("target_answer", ""),
        "contrast_answer": row.get("contrast_answer", ""),
        "target_evidence_title": row.get("target_evidence_title", ""),
        "contrast_evidence_title": row.get("contrast_evidence_title", ""),
        "target_evidence_url": row.get("target_evidence_url", ""),
        "contrast_evidence_url": row.get("contrast_evidence_url", ""),
        "evidence_hint": row.get("evidence_hint", ""),
        "score": score,
        "risk_band": band,
        "flags": " | ".join(dict.fromkeys(flags)),
        "notes": " ; ".join(notes),
        "target_answer_type": target_type,
        "contrast_answer_type": contrast_type,
        "relation_target_strong": row.get("relation_target_strong", ""),
        "relation_contrast_strong": row.get("relation_contrast_strong", ""),
    }


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
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--fetches", type=Path, default=DEFAULT_FETCHES)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    rows = [row for row in load_jsonl(args.input) if row.get("split_type") == "ambiguous"]
    fetches = load_fetches(args.fetches)
    audited = [audit_row(row, fetches) for row in rows]
    audited.sort(key=lambda row: (-int(row["score"]), row["country"], row["id"]))

    flagged = [row for row in audited if row["risk_band"] != "clean"]
    medium_plus = [row for row in audited if row["risk_band"] in {"high", "medium"}]
    high = [row for row in audited if row["risk_band"] == "high"]

    band_counts = Counter(row["risk_band"] for row in audited)
    flag_counts = Counter()
    country_counts: dict[str, Counter[str]] = defaultdict(Counter)
    topic_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in audited:
        band = row["risk_band"]
        country_counts[row["country"]][band] += 1
        topic_counts[row["topic"]][band] += 1
        for flag in str(row["flags"]).split(" | "):
            if flag:
                flag_counts[flag] += 1

    calibration_ids = {
        "localnewsqa_ambig_semantic_gold_0210",
        "localnewsqa_ambig_semantic_gold_0623",
        "localnewsqa_ambig_semantic_gold_0818",
    }
    calibration = [row for row in audited if row["id"] in calibration_ids]

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "weak_locale_pattern_audit_all_rows.csv", audited)
    write_csv(args.outdir / "weak_locale_pattern_audit_flagged_rows.csv", flagged)
    write_csv(args.outdir / "weak_locale_pattern_audit_medium_plus_rows.csv", medium_plus)
    write_csv(args.outdir / "weak_locale_pattern_audit_high_rows.csv", high)
    write_csv(args.outdir / "weak_locale_pattern_audit_calibration_rows.csv", calibration)

    summary = {
        "input": str(args.input),
        "fetches": str(args.fetches),
        "outdir": str(args.outdir),
        "rows_audited": len(audited),
        "band_counts": dict(band_counts),
        "flag_counts": dict(flag_counts.most_common()),
        "medium_plus_count": len(medium_plus),
        "high_count": len(high),
        "calibration_ids": sorted(calibration_ids),
        "calibration_rows": calibration,
        "country_band_counts": {k: dict(v) for k, v in sorted(country_counts.items())},
        "topic_band_counts": {k: dict(v) for k, v in sorted(topic_counts.items())},
        "outputs": {
            "all_rows": str(args.outdir / "weak_locale_pattern_audit_all_rows.csv"),
            "flagged_rows": str(args.outdir / "weak_locale_pattern_audit_flagged_rows.csv"),
            "medium_plus_rows": str(args.outdir / "weak_locale_pattern_audit_medium_plus_rows.csv"),
            "high_rows": str(args.outdir / "weak_locale_pattern_audit_high_rows.csv"),
            "calibration_rows": str(args.outdir / "weak_locale_pattern_audit_calibration_rows.csv"),
            "summary": str(args.outdir / "weak_locale_pattern_audit_summary.json"),
        },
    }
    (args.outdir / "weak_locale_pattern_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
