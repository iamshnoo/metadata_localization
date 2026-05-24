#!/usr/bin/env python3

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


GENERIC_OPTION_PATTERNS = [
    re.compile(r"^all of the above$", re.IGNORECASE),
    re.compile(r"^none of the above$", re.IGNORECASE),
    re.compile(r"^both a and b$", re.IGNORECASE),
    re.compile(r"^cannot be determined$", re.IGNORECASE),
    re.compile(r"^not enough information$", re.IGNORECASE),
]

LOW_INFORMATION_PATTERNS = [
    re.compile(r"\busually named\b", re.IGNORECASE),
    re.compile(r"\bcommonly named\b", re.IGNORECASE),
    re.compile(r"\bwould usually be named\b", re.IGNORECASE),
    re.compile(r"\bmost likely to be cited\b", re.IGNORECASE),
    re.compile(r"\bin stories about\b", re.IGNORECASE),
    re.compile(r"\bin similar coverage\b", re.IGNORECASE),
    re.compile(r"\bmore likely use\b", re.IGNORECASE),
    re.compile(r"\bmore common elsewhere\b", re.IGNORECASE),
    re.compile(r"\bcounterpart\b", re.IGNORECASE),
    re.compile(r"\boften involved when\b", re.IGNORECASE),
    re.compile(r"\boften treated\b", re.IGNORECASE),
]


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return text


def country_name_variants(name: str) -> Set[str]:
    variants = set()
    raw = (name or "").strip()
    if not raw:
        return variants
    variants.add(raw.lower())
    variants.add(normalize_text(raw))
    if raw == "United States":
        variants.update({"united states", "usa", "u.s.", "u.s", "us", "america"})
    elif raw == "United Kingdom":
        variants.update({"united kingdom", "uk", "u.k.", "u.k", "britain", "great britain"})
    return {v for v in variants if v}


def answer_leaks_into_question(question: str, answer: str) -> bool:
    q_norm = normalize_text(question)
    a_norm = normalize_text(answer)
    return len(a_norm) >= 4 and a_norm in q_norm


def has_generic_option(options: Iterable[str]) -> bool:
    for option in options:
        text = (option or "").strip()
        for pattern in GENERIC_OPTION_PATTERNS:
            if pattern.search(text):
                return True
    return False


def is_low_information_question(question: str) -> bool:
    text = (question or "").strip()
    return any(pattern.search(text) for pattern in LOW_INFORMATION_PATTERNS)


def option_set_without_correct(options: Iterable[str], correct_answer: str) -> Set[str]:
    correct_norm = normalize_text(correct_answer)
    return {
        normalize_text(option)
        for option in options
        if normalize_text(option) and normalize_text(option) != correct_norm
    }


def find_rejection_reasons(item: Dict[str, Any], seen_questions: Set[Tuple[str, str, str]]) -> List[str]:
    reasons: List[str] = []

    question = str(item.get("question") or "").strip()
    correct_answer = str(item.get("correct_answer") or "").strip()
    split_type = str(item.get("split_type") or "").strip()
    country = str(item.get("country") or "").strip()
    contrast_country = str(item.get("contrast_country") or "").strip()
    target_answer = str(item.get("target_answer") or "").strip()
    contrast_answer = str(item.get("contrast_answer") or "").strip()
    options = item.get("options")
    distractors = item.get("distractors")
    year = item.get("year")

    norm_question = normalize_text(question)
    duplicate_key = (country, split_type, norm_question)
    if duplicate_key in seen_questions:
        reasons.append("duplicate_normalized_question")
    else:
        seen_questions.add(duplicate_key)

    if not question:
        reasons.append("format_missing_question")
    if not correct_answer:
        reasons.append("format_missing_correct_answer")

    if not isinstance(options, list) or len(options) != 4:
        reasons.append("format_invalid_options")
    if not isinstance(distractors, list) or len(distractors) != 3:
        reasons.append("format_invalid_distractors")

    if isinstance(options, list):
        option_norms = [normalize_text(str(option)) for option in options]
        if any(not option for option in option_norms):
            reasons.append("format_empty_option")
        if len(set(option_norms)) != len(option_norms):
            reasons.append("format_duplicate_options")
        if normalize_text(correct_answer) not in option_norms:
            reasons.append("format_correct_not_in_options")

    if isinstance(distractors, list):
        distractor_norms = [normalize_text(str(option)) for option in distractors]
        if any(not option for option in distractor_norms):
            reasons.append("format_empty_distractor")
        if normalize_text(correct_answer) in distractor_norms:
            reasons.append("weak_distractor_contains_correct_answer")
        if len(set(distractor_norms)) != len(distractor_norms):
            reasons.append("weak_duplicate_distractors")
        if isinstance(options, list) and len(options) == 4:
            if set(distractor_norms) != option_set_without_correct(options, correct_answer):
                reasons.append("format_distractors_do_not_match_options")

    if not isinstance(year, int) or year < 2010 or year > 2024:
        reasons.append("format_invalid_year")

    if isinstance(options, list) and has_generic_option(str(option) for option in options):
        reasons.append("weak_generic_option")

    if answer_leaks_into_question(question, correct_answer):
        reasons.append("answer_leakage")

    if split_type == "explicit":
        if item.get("ambiguity_flag") is not False:
            reasons.append("ambiguity_explicit_flag_not_false")
    elif split_type == "ambiguous":
        if item.get("ambiguity_flag") is not True:
            reasons.append("ambiguity_ambiguous_flag_not_true")
        if not target_answer:
            reasons.append("ambiguity_missing_target_answer")
        if not contrast_answer:
            reasons.append("ambiguity_missing_contrast_answer")
        if correct_answer != target_answer:
            reasons.append("ambiguity_correct_answer_mismatch")
        if target_answer and contrast_answer and normalize_text(target_answer) == normalize_text(contrast_answer):
            reasons.append("ambiguity_target_equals_contrast")
        if isinstance(options, list):
            option_norms = {normalize_text(str(option)) for option in options}
            if target_answer and normalize_text(target_answer) not in option_norms:
                reasons.append("ambiguity_target_not_in_options")
            if contrast_answer and normalize_text(contrast_answer) not in option_norms:
                reasons.append("ambiguity_contrast_not_in_options")
        question_lower = question.lower()
        for variant in country_name_variants(country) | country_name_variants(contrast_country):
            if variant and variant in question_lower:
                reasons.append("ambiguity_question_mentions_country")
                break

    if is_low_information_question(question):
        reasons.append("low_information_question")

    return sorted(set(reasons))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune LocalNewsQA-Core generation candidates before evidence retrieval.")
    parser.add_argument("--in-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--rejects-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    args = parser.parse_args()

    in_path = Path(args.in_jsonl)
    kept_path = Path(args.out_jsonl)
    rejects_path = Path(args.rejects_jsonl)
    report_path = Path(args.report_json)

    seen_questions: Set[Tuple[str, str, str]] = set()
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    kept_bucket_counts: Counter[Tuple[str, str]] = Counter()

    with in_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            reasons = find_rejection_reasons(item, seen_questions)
            if reasons:
                rejected_row = dict(item)
                rejected_row["rejection_reasons"] = reasons
                rejected_row["source_line"] = line_no
                rejected.append(rejected_row)
                reason_counts.update(reasons)
                for reason in reasons:
                    category_counts[reason.split("_", 1)[0]] += 1
            else:
                kept.append(item)
                kept_bucket_counts[(item["country"], item["split_type"])] += 1

    write_jsonl(kept_path, kept)
    write_jsonl(rejects_path, rejected)

    bucket_counts = [
        {"country": country, "split_type": split_type, "count": count}
        for (country, split_type), count in sorted(kept_bucket_counts.items())
    ]

    report = {
        "input_path": str(in_path),
        "kept_path": str(kept_path),
        "rejects_path": str(rejects_path),
        "input_count": len(kept) + len(rejected),
        "kept_count": len(kept),
        "rejected_count": len(rejected),
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "rejection_category_counts": dict(sorted(category_counts.items())),
        "kept_bucket_counts": bucket_counts,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)

    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
