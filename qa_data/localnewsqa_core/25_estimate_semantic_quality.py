#!/usr/bin/env python3

import argparse
import csv
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "qa_data/localnewsqa_core/runs/human_validation_full_35874_web_certified_curated.csv"
)
DEFAULT_OUTDIR = ROOT / "qa_data/localnewsqa_core/runs/quality_audit"

COUNTRY_MARKERS = {
    "Bangladesh": ["bangladesh", "bangladeshi"],
    "Canada": ["canada", "canadian"],
    "Ghana": ["ghana", "ghanaian"],
    "Hong Kong": ["hong kong"],
    "India": ["india", "indian"],
    "Ireland": ["ireland", "irish"],
    "Jamaica": ["jamaica", "jamaican"],
    "Kenya": ["kenya", "kenyan"],
    "Malaysia": ["malaysia", "malaysian"],
    "Nigeria": ["nigeria", "nigerian"],
    "Pakistan": ["pakistan", "pakistani"],
    "Philippines": ["philippines", "philippine", "filipino"],
    "South Africa": ["south africa", "south african"],
    "Sri Lanka": ["sri lanka", "sri lankan"],
    "Tanzania": ["tanzania", "tanzanian"],
    "United Kingdom": ["united kingdom", "uk", "u k", "britain", "british", "england"],
    "United States": ["united states", "usa", "u s", "u.s", "american"],
}

STOPWORDS = {
    "a",
    "about",
    "after",
    "an",
    "and",
    "annual",
    "are",
    "as",
    "at",
    "be",
    "by",
    "called",
    "country",
    "date",
    "day",
    "did",
    "does",
    "during",
    "each",
    "for",
    "from",
    "in",
    "is",
    "it",
    "its",
    "local",
    "most",
    "national",
    "news",
    "of",
    "on",
    "or",
    "report",
    "reports",
    "story",
    "that",
    "the",
    "their",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "would",
}

ANSWER_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "de",
    "for",
    "in",
    "la",
    "of",
    "on",
    "the",
    "to",
}

GENERIC_TITLES = {
    "association football",
    "athletics",
    "civil service",
    "constitution day",
    "county clerk",
    "court",
    "cricket",
    "democracy day",
    "election day",
    "field hockey",
    "football",
    "golf",
    "hockey",
    "independence day",
    "labour day",
    "national day",
    "parliament",
    "primary school",
    "republic day",
    "secretary of state",
    "supreme court",
    "union day",
}

MONTH_TO_NUM = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

BLACKLISTED_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "m.reddit.com",
    "reddit.com",
    "tiktok.com",
    "twitter.com",
    "www.facebook.com",
    "www.instagram.com",
    "www.linkedin.com",
    "www.reddit.com",
    "www.tiktok.com",
    "www.twitter.com",
    "www.x.com",
    "x.com",
    "youtube.com",
    "www.youtube.com",
}

CONSERVATIVE_WEIGHTS = {
    "manual_reject": 1.00,
    "missing_certification": 0.00,
    "missing_expected_answer": 1.00,
    "answer_not_in_options": 0.95,
    "option_count_not_4": 0.95,
    "duplicate_options": 0.75,
    "ambiguous_same_answer_canonical": 0.90,
    "ambiguous_same_answer_date": 0.90,
    "ambiguous_same_answer_number": 0.90,
    "ambiguous_same_answer_acronym": 0.80,
    "ambiguous_near_duplicate_answer": 0.35,
    "ambiguous_high_token_overlap_answer": 0.15,
    "generic_date_page_for_holiday_claim": 0.06,
    "generic_number_page_without_context": 0.08,
    "generic_page_without_claim_context": 0.00,
    "generic_title_without_locale": 0.00,
    "option_locale_marker_leakage": 0.15,
    "ambiguous_answer_exact_leakage": 0.25,
    "ambiguous_answer_token_leakage": 0.10,
    "ambiguous_locale_leakage_in_question": 0.08,
    "explicit_missing_locale_in_question": 0.01,
    "answer_absent_from_evidence_text": 0.08,
    "locale_absent_from_evidence_text": 0.00,
    "boilerplate_excerpt_only": 0.00,
    "blacklisted_domain": 0.00,
}

EVIDENCE_SUPPORT_WEIGHTS = {
    "manual_reject": 0.00,
    "missing_certification": 0.85,
    "missing_expected_answer": 0.00,
    "answer_not_in_options": 0.00,
    "option_count_not_4": 0.00,
    "duplicate_options": 0.00,
    "ambiguous_same_answer_canonical": 0.00,
    "ambiguous_same_answer_date": 0.00,
    "ambiguous_same_answer_number": 0.00,
    "ambiguous_same_answer_acronym": 0.00,
    "ambiguous_near_duplicate_answer": 0.00,
    "ambiguous_high_token_overlap_answer": 0.00,
    "generic_date_page_for_holiday_claim": 0.50,
    "generic_number_page_without_context": 0.45,
    "generic_page_without_claim_context": 0.20,
    "generic_title_without_locale": 0.12,
    "option_locale_marker_leakage": 0.00,
    "ambiguous_answer_exact_leakage": 0.00,
    "ambiguous_answer_token_leakage": 0.00,
    "ambiguous_locale_leakage_in_question": 0.00,
    "explicit_missing_locale_in_question": 0.00,
    "answer_absent_from_evidence_text": 0.20,
    "locale_absent_from_evidence_text": 0.08,
    "boilerplate_excerpt_only": 0.03,
    "blacklisted_domain": 0.50,
}

REVIEW_WEIGHTS = {
    "manual_reject": 1.00,
    "missing_certification": 1.00,
    "missing_expected_answer": 1.00,
    "answer_not_in_options": 1.00,
    "option_count_not_4": 1.00,
    "duplicate_options": 0.95,
    "ambiguous_same_answer_canonical": 0.95,
    "ambiguous_same_answer_date": 0.95,
    "ambiguous_same_answer_number": 0.95,
    "ambiguous_same_answer_acronym": 0.90,
    "ambiguous_near_duplicate_answer": 0.70,
    "ambiguous_high_token_overlap_answer": 0.55,
    "generic_date_page_for_holiday_claim": 0.70,
    "generic_number_page_without_context": 0.65,
    "generic_page_without_claim_context": 0.55,
    "generic_title_without_locale": 0.45,
    "option_locale_marker_leakage": 0.45,
    "ambiguous_answer_exact_leakage": 0.50,
    "ambiguous_answer_token_leakage": 0.35,
    "ambiguous_locale_leakage_in_question": 0.35,
    "explicit_missing_locale_in_question": 0.25,
    "answer_absent_from_evidence_text": 0.30,
    "locale_absent_from_evidence_text": 0.20,
    "boilerplate_excerpt_only": 0.05,
    "blacklisted_domain": 0.50,
}

ISSUE_FAMILIES = {
    "manual_reject": "known_bad",
    "missing_certification": "certification",
    "missing_expected_answer": "structure",
    "answer_not_in_options": "structure",
    "option_count_not_4": "structure",
    "duplicate_options": "structure",
    "ambiguous_same_answer_canonical": "answer_equivalence",
    "ambiguous_same_answer_date": "answer_equivalence",
    "ambiguous_same_answer_number": "answer_equivalence",
    "ambiguous_same_answer_acronym": "answer_equivalence",
    "ambiguous_near_duplicate_answer": "answer_equivalence",
    "ambiguous_high_token_overlap_answer": "answer_equivalence",
    "generic_date_page_for_holiday_claim": "evidence_support",
    "generic_number_page_without_context": "evidence_support",
    "generic_page_without_claim_context": "evidence_support",
    "generic_title_without_locale": "evidence_support",
    "answer_absent_from_evidence_text": "evidence_support",
    "locale_absent_from_evidence_text": "evidence_support",
    "boilerplate_excerpt_only": "evidence_display",
    "blacklisted_domain": "evidence_source",
    "option_locale_marker_leakage": "leakage",
    "ambiguous_answer_exact_leakage": "leakage",
    "ambiguous_answer_token_leakage": "leakage",
    "ambiguous_locale_leakage_in_question": "leakage",
    "explicit_missing_locale_in_question": "question_framing",
}

LOWER_BOUND_SEMANTIC_ISSUES = {
    "manual_reject",
    "missing_expected_answer",
    "answer_not_in_options",
    "option_count_not_4",
    "duplicate_options",
    "ambiguous_same_answer_canonical",
    "ambiguous_same_answer_date",
    "ambiguous_same_answer_number",
    "ambiguous_same_answer_acronym",
}


def normalize(text: str) -> str:
    text = html.unescape(str(text or ""))
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9.%]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str, *, stopwords: set[str] = STOPWORDS) -> list[str]:
    return [token for token in normalize(text).split() if token and token not in stopwords]


def canonical_answer_text(answer: str) -> str:
    return " ".join(tokens(answer, stopwords=ANSWER_STOPWORDS))


def answer_terms(answer: str) -> set[str]:
    return {tok for tok in tokens(answer, stopwords=ANSWER_STOPWORDS) if len(tok) > 1}


def split_type(row: dict) -> str:
    return row.get("split") or ("ambiguous" if "ambig" in row.get("id", "") else "explicit")


def option_parts(options: str) -> list[str]:
    parts = [part.strip() for part in str(options or "").split("||") if part.strip()]
    cleaned = []
    for part in parts:
        if re.match(r"^[A-D]:", part):
            part = part.split(":", 1)[1].strip()
        cleaned.append(part)
    return cleaned


def domain_of(url: str) -> str:
    return (urlparse(str(url or "")).netloc or "").lower()


def url_slug(url: str) -> str:
    path = unquote(urlparse(str(url or "")).path or "")
    return path.rsplit("/", 1)[-1].replace("_", " ")


def evidence_title(row: dict, side: str) -> str:
    return row.get(f"{side}_evidence_title") or url_slug(row.get(f"{side}_evidence_url", ""))


def evidence_text(row: dict, side: str) -> str:
    return normalize(
        " ".join(
            [
                row.get(f"{side}_evidence_title", ""),
                row.get(f"{side}_evidence_snippet", ""),
                row.get(f"{side}_evidence_excerpt", ""),
                url_slug(row.get(f"{side}_evidence_url", "")),
            ]
        )
    )


def has_country_marker(text: str, country: str) -> bool:
    norm = normalize(text)
    return any(normalize(marker) in norm for marker in COUNTRY_MARKERS.get(country, [country]))


def option_has_locale_marker(options: str, country: str) -> bool:
    return any(has_country_marker(option, country) for option in option_parts(options))


def answer_in_options(answer: str, options: str) -> bool:
    answer_norm = canonical_answer_text(answer)
    return bool(answer_norm) and answer_norm in {
        canonical_answer_text(part) for part in option_parts(options)
    }


def canonical_date(text: str) -> str:
    norm = normalize(text).replace(".", "")
    norm = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", norm)
    parts = norm.split()
    if len(parts) != 2:
        return ""
    if parts[0].isdigit() and parts[1] in MONTH_TO_NUM:
        day = int(parts[0])
        month = MONTH_TO_NUM[parts[1]]
    elif parts[1].isdigit() and parts[0] in MONTH_TO_NUM:
        day = int(parts[1])
        month = MONTH_TO_NUM[parts[0]]
    else:
        return ""
    if not 1 <= day <= 31:
        return ""
    return f"{month:02d}-{day:02d}"


def canonical_number(text: str) -> str:
    norm = normalize(text)
    norm = norm.replace("percent", "%").replace("per cent", "%")
    if re.fullmatch(r"\d+(\.\d+)?%?", norm):
        return norm
    if norm in NUMBER_WORDS:
        return str(NUMBER_WORDS[norm])
    pieces = norm.split()
    if len(pieces) == 2 and pieces[0] in NUMBER_WORDS and pieces[1] in NUMBER_WORDS:
        return str(NUMBER_WORDS[pieces[0]] + NUMBER_WORDS[pieces[1]])
    return ""


def initials(text: str) -> str:
    toks = [tok for tok in tokens(text, stopwords=ANSWER_STOPWORDS) if tok]
    if len(toks) < 2:
        return ""
    return "".join(tok[0] for tok in toks)


def is_generic_title(title: str) -> bool:
    canon = canonical_answer_text(title)
    return canon in GENERIC_TITLES or bool(canonical_date(title)) or bool(canonical_number(title))


def claim_context_terms(row: dict, side: str) -> set[str]:
    answer = row.get(f"{side}_answer", "")
    country = row.get(f"{side}_country", "")
    excluded = answer_terms(answer)
    for marker in COUNTRY_MARKERS.get(country, [country]):
        excluded |= set(tokens(marker))
    raw = " ".join([row.get("question", ""), row.get("evidence_hint", ""), row.get("topic", "")])
    terms = {
        tok
        for tok in tokens(raw)
        if len(tok) >= 4 and tok not in excluded and not tok.isdigit()
    }
    return terms


def context_supported(row: dict, side: str) -> bool:
    terms = claim_context_terms(row, side)
    if not terms:
        return True
    text_terms = set(evidence_text(row, side).split())
    hits = terms & text_terms
    return len(hits) >= min(2, len(terms))


def answer_equivalence_issues(target: str, contrast: str) -> list[str]:
    issues = []
    target_canon = canonical_answer_text(target)
    contrast_canon = canonical_answer_text(contrast)
    if not target_canon or not contrast_canon:
        return issues
    if target_canon == contrast_canon:
        issues.append("ambiguous_same_answer_canonical")
        return issues

    target_date = canonical_date(target)
    contrast_date = canonical_date(contrast)
    if target_date and target_date == contrast_date:
        issues.append("ambiguous_same_answer_date")
        return issues

    target_num = canonical_number(target)
    contrast_num = canonical_number(contrast)
    if target_num and target_num == contrast_num:
        issues.append("ambiguous_same_answer_number")
        return issues

    target_initials = initials(target)
    contrast_initials = initials(contrast)
    if target_canon == contrast_initials or contrast_canon == target_initials:
        issues.append("ambiguous_same_answer_acronym")
        return issues

    ratio = SequenceMatcher(None, target_canon, contrast_canon).ratio()
    target_terms = answer_terms(target)
    contrast_terms = answer_terms(contrast)
    if ratio >= 0.88:
        issues.append("ambiguous_near_duplicate_answer")
    elif target_terms and contrast_terms:
        overlap = len(target_terms & contrast_terms)
        smaller = min(len(target_terms), len(contrast_terms))
        jaccard = overlap / len(target_terms | contrast_terms)
        if smaller >= 2 and (overlap / smaller >= 0.75 or jaccard >= 0.60):
            issues.append("ambiguous_high_token_overlap_answer")
    return issues


def question_leakage_issues(row: dict) -> list[str]:
    issues = []
    question = row.get("question", "")
    q_norm = normalize(question)
    for side in ("target", "contrast"):
        answer = row.get(f"{side}_answer", "")
        answer_norm = normalize(answer)
        if not answer_norm:
            continue
        if answer_norm in q_norm:
            issues.append("ambiguous_answer_exact_leakage")
            continue
        terms = answer_terms(answer)
        if len(terms) >= 2 and terms.issubset(set(q_norm.split())):
            issues.append("ambiguous_answer_token_leakage")
    if has_country_marker(question, row.get("target_country", "")) or has_country_marker(
        question, row.get("contrast_country", "")
    ):
        issues.append("ambiguous_locale_leakage_in_question")
    return sorted(set(issues))


def evidence_issues(row: dict, side: str) -> list[str]:
    issues = []
    answer = row.get(f"{side}_answer", "")
    country = row.get(f"{side}_country", "")
    if not answer.strip():
        return issues

    url = row.get(f"{side}_evidence_url", "")
    title = evidence_title(row, side)
    title_canon = canonical_answer_text(title)
    answer_canon = canonical_answer_text(answer)
    text = evidence_text(row, side)
    snippet = row.get(f"{side}_evidence_snippet", "")
    excerpt = row.get(f"{side}_evidence_excerpt", "")
    has_boilerplate_excerpt = excerpt and "jump to content" in normalize(excerpt)

    if row.get(f"{side}_source_certified", "") != "yes":
        issues.append("missing_certification")
    if url and domain_of(url) in BLACKLISTED_DOMAINS:
        issues.append("blacklisted_domain")
    if has_boilerplate_excerpt and not snippet.strip():
        issues.append("boilerplate_excerpt_only")

    answer_set = answer_terms(answer)
    if answer_set and not answer_set.issubset(set(text.split())):
        issues.append("answer_absent_from_evidence_text")
    if country and not has_country_marker(text, country):
        issues.append("locale_absent_from_evidence_text")

    generic = is_generic_title(title)
    no_context = not context_supported(row, side)
    title_is_answer = bool(title_canon and answer_canon and title_canon == answer_canon)

    if (
        row.get("topic") == "Public life and holidays"
        and canonical_date(answer)
        and canonical_date(title)
        and no_context
    ):
        issues.append("generic_date_page_for_holiday_claim")
    if canonical_number(answer) and canonical_number(title) and no_context:
        issues.append("generic_number_page_without_context")
    if generic and country and not has_country_marker(text, country):
        issues.append("generic_title_without_locale")
    if (generic or title_is_answer) and no_context:
        issues.append("generic_page_without_claim_context")
    return sorted(set(issues))


def structural_issues(row: dict) -> list[str]:
    issues = []
    split = split_type(row)
    options = row.get("options", "")
    parts = option_parts(options)
    if len(parts) != 4:
        issues.append("option_count_not_4")
    if len({canonical_answer_text(part) for part in parts}) != len(parts):
        issues.append("duplicate_options")
    if row.get("manual_review_status") == "manual_reject":
        issues.append("manual_reject")
    if not row.get("target_answer", "").strip():
        issues.append("missing_expected_answer")
    elif not answer_in_options(row.get("target_answer", ""), options):
        issues.append("answer_not_in_options")
    if split == "ambiguous":
        if not row.get("contrast_answer", "").strip():
            issues.append("missing_expected_answer")
        elif not answer_in_options(row.get("contrast_answer", ""), options):
            issues.append("answer_not_in_options")
        issues.extend(answer_equivalence_issues(row.get("target_answer", ""), row.get("contrast_answer", "")))
        issues.extend(question_leakage_issues(row))
        if option_has_locale_marker(options, row.get("target_country", "")) or option_has_locale_marker(
            options, row.get("contrast_country", "")
        ):
            issues.append("option_locale_marker_leakage")
    elif row.get("target_country") and not has_country_marker(row.get("question", ""), row.get("target_country", "")):
        issues.append("explicit_missing_locale_in_question")
    return sorted(set(issues))


def row_issue_records(row: dict) -> list[dict]:
    records = []
    for issue in structural_issues(row):
        records.append({"side": "row", "issue": issue})
    for side in ("target", "contrast"):
        if side == "contrast" and split_type(row) == "explicit":
            continue
        for issue in evidence_issues(row, side):
            records.append({"side": side, "issue": issue})
    return records


def combined_probability(issues: list[str], weights: dict[str, float], *, default: float = 0.0) -> float:
    if not issues:
        return 0.0
    prob_clean = 1.0
    for issue in issues:
        prob_clean *= 1.0 - weights.get(issue, default)
    return round(1.0 - prob_clean, 4)


def semantic_risk_band(prob: float) -> str:
    if prob >= 0.90:
        return "hard_or_near_hard"
    if prob >= 0.50:
        return "high"
    if prob >= 0.20:
        return "medium"
    if prob > 0:
        return "low"
    return "clean"


def review_priority_band(prob: float) -> str:
    if prob >= 0.90:
        return "hard_or_near_hard"
    if prob >= 0.70:
        return "high"
    if prob >= 0.40:
        return "medium"
    if prob > 0:
        return "low"
    return "clean"


def evidence_support_band(prob: float) -> str:
    if prob >= 0.90:
        return "hard_or_near_hard"
    if prob >= 0.60:
        return "high"
    if prob >= 0.30:
        return "medium"
    if prob > 0:
        return "low"
    return "clean"


def output_base(row: dict) -> dict:
    return {
        "id": row.get("id", ""),
        "split": split_type(row),
        "country": row.get("country", ""),
        "continent": row.get("continent", ""),
        "topic": row.get("topic", ""),
        "year": row.get("year", ""),
        "target_country": row.get("target_country", ""),
        "contrast_country": row.get("contrast_country", ""),
        "target_answer": row.get("target_answer", ""),
        "contrast_answer": row.get("contrast_answer", ""),
        "question": row.get("question", ""),
        "options": row.get("options", ""),
        "evidence_hint": row.get("evidence_hint", ""),
        "target_evidence_url": row.get("target_evidence_url", ""),
        "target_evidence_title": row.get("target_evidence_title", ""),
        "target_evidence_snippet": row.get("target_evidence_snippet", ""),
        "target_match_type": row.get("target_match_type", ""),
        "target_source_certified": row.get("target_source_certified", ""),
        "contrast_evidence_url": row.get("contrast_evidence_url", ""),
        "contrast_evidence_title": row.get("contrast_evidence_title", ""),
        "contrast_evidence_snippet": row.get("contrast_evidence_snippet", ""),
        "contrast_match_type": row.get("contrast_match_type", ""),
        "contrast_source_certified": row.get("contrast_source_certified", ""),
        "manual_review_status": row.get("manual_review_status", ""),
        "manual_review_reason": row.get("manual_review_reason", ""),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(score_rows: list[dict], issue_rows: list[dict]) -> dict:
    semantic_band_counts = Counter(row["semantic_risk_band"] for row in score_rows)
    evidence_band_counts = Counter(row["evidence_support_band"] for row in score_rows)
    review_band_counts = Counter(row["review_priority_band"] for row in score_rows)
    issue_occurrence_counts = Counter(row["issue"] for row in issue_rows)
    family_occurrence_counts = Counter(row["family"] for row in issue_rows)
    issue_row_counts = defaultdict(set)
    family_row_counts = defaultdict(set)
    semantic_split_band_counts = defaultdict(Counter)
    semantic_topic_band_counts = defaultdict(Counter)
    evidence_split_band_counts = defaultdict(Counter)
    evidence_topic_band_counts = defaultdict(Counter)
    review_split_band_counts = defaultdict(Counter)
    review_topic_band_counts = defaultdict(Counter)
    lower_bound_row_ids = set()

    for row in score_rows:
        semantic_split_band_counts[row["split"]][row["semantic_risk_band"]] += 1
        semantic_topic_band_counts[row["topic"]][row["semantic_risk_band"]] += 1
        evidence_split_band_counts[row["split"]][row["evidence_support_band"]] += 1
        evidence_topic_band_counts[row["topic"]][row["evidence_support_band"]] += 1
        review_split_band_counts[row["split"]][row["review_priority_band"]] += 1
        review_topic_band_counts[row["topic"]][row["review_priority_band"]] += 1
        if row["has_lower_bound_semantic_issue"] == "yes":
            lower_bound_row_ids.add(row["id"])

    for row in issue_rows:
        issue_row_counts[row["issue"]].add(row["id"])
        family_row_counts[row["family"]].add(row["id"])

    conservative_expected_errors = sum(float(row["semantic_error_probability"]) for row in score_rows)
    evidence_support_proxy = sum(float(row["evidence_support_score"]) for row in score_rows)
    review_priority_proxy = sum(float(row["review_priority_score"]) for row in score_rows)
    return {
        "rows": len(score_rows),
        "semantic_lower_bound_rows": len(lower_bound_row_ids),
        "semantic_lower_bound_rate": round(len(lower_bound_row_ids) / len(score_rows), 4),
        "conservative_expected_semantic_error_count": round(conservative_expected_errors, 1),
        "conservative_expected_semantic_error_rate": round(
            conservative_expected_errors / len(score_rows), 4
        ),
        "evidence_support_expected_count_proxy": round(evidence_support_proxy, 1),
        "evidence_support_expected_rate_proxy": round(evidence_support_proxy / len(score_rows), 4),
        "review_priority_expected_count_proxy": round(review_priority_proxy, 1),
        "review_priority_expected_rate_proxy": round(review_priority_proxy / len(score_rows), 4),
        "semantic_risk_band_counts": dict(semantic_band_counts),
        "evidence_support_band_counts": dict(evidence_band_counts),
        "review_priority_band_counts": dict(review_band_counts),
        "issue_occurrence_counts": dict(issue_occurrence_counts.most_common()),
        "issue_row_counts": {
            issue: len(row_ids)
            for issue, row_ids in sorted(
                issue_row_counts.items(), key=lambda item: (-len(item[1]), item[0])
            )
        },
        "issue_family_occurrence_counts": dict(family_occurrence_counts.most_common()),
        "issue_family_row_counts": {
            family: len(row_ids)
            for family, row_ids in sorted(
                family_row_counts.items(), key=lambda item: (-len(item[1]), item[0])
            )
        },
        "by_split_semantic_risk_band_counts": {
            split: dict(counter) for split, counter in sorted(semantic_split_band_counts.items())
        },
        "by_topic_semantic_risk_band_counts": {
            topic: dict(counter) for topic, counter in sorted(semantic_topic_band_counts.items())
        },
        "by_split_evidence_support_band_counts": {
            split: dict(counter) for split, counter in sorted(evidence_split_band_counts.items())
        },
        "by_topic_evidence_support_band_counts": {
            topic: dict(counter) for topic, counter in sorted(evidence_topic_band_counts.items())
        },
        "by_split_review_priority_band_counts": {
            split: dict(counter) for split, counter in sorted(review_split_band_counts.items())
        },
        "by_topic_review_priority_band_counts": {
            topic: dict(counter) for topic, counter in sorted(review_topic_band_counts.items())
        },
        "lower_bound_semantic_issues": sorted(LOWER_BOUND_SEMANTIC_ISSUES),
        "conservative_weights": CONSERVATIVE_WEIGHTS,
        "evidence_support_weights": EVIDENCE_SUPPORT_WEIGHTS,
        "review_weights": REVIEW_WEIGHTS,
        "notes": [
            "semantic_lower_bound_rows is an exact minimum count of rows with direct structural or answer-equivalence defects.",
            "conservative_expected_semantic_error_count is a heuristic noisy-or estimate, intended as a cautious semantic error estimate rather than a calibrated human label rate.",
            "evidence_support_expected_count_proxy measures how often the provided evidence looks weak or generic; it is not a semantic error estimate.",
            "review_priority_expected_count_proxy is not an error-rate estimate; it is an aggressive prioritization score for manual review ordering.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate semantic quality risks for LocalNewsQA.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--prefix", default="semantic_quality_full_35874")
    args = parser.parse_args()

    with args.input.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    score_rows = []
    issue_rows = []
    for row in rows:
        issue_records = row_issue_records(row)
        issues = [record["issue"] for record in issue_records]
        semantic_probability = combined_probability(issues, CONSERVATIVE_WEIGHTS, default=0.03)
        evidence_support_probability = combined_probability(issues, EVIDENCE_SUPPORT_WEIGHTS, default=0.0)
        review_probability = combined_probability(issues, REVIEW_WEIGHTS, default=0.10)
        lower_bound_issues = sorted({issue for issue in issues if issue in LOWER_BOUND_SEMANTIC_ISSUES})
        base = output_base(row)
        score_row = {
            **base,
            "semantic_error_probability": f"{semantic_probability:.4f}",
            "semantic_risk_band": semantic_risk_band(semantic_probability),
            "evidence_support_score": f"{evidence_support_probability:.4f}",
            "evidence_support_band": evidence_support_band(evidence_support_probability),
            "review_priority_score": f"{review_probability:.4f}",
            "review_priority_band": review_priority_band(review_probability),
            "has_lower_bound_semantic_issue": "yes" if lower_bound_issues else "no",
            "lower_bound_semantic_issues": " | ".join(lower_bound_issues),
            "issues": " | ".join(f"{record['side']}:{record['issue']}" for record in issue_records),
            "issue_families": " | ".join(
                sorted({ISSUE_FAMILIES.get(record["issue"], "other") for record in issue_records})
            ),
        }
        score_rows.append(score_row)
        for record in issue_records:
            issue_rows.append(
                {
                    **base,
                    "side": record["side"],
                    "issue": record["issue"],
                    "family": ISSUE_FAMILIES.get(record["issue"], "other"),
                    "conservative_weight": CONSERVATIVE_WEIGHTS.get(record["issue"], 0.03),
                    "evidence_support_weight": EVIDENCE_SUPPORT_WEIGHTS.get(record["issue"], 0.0),
                    "review_weight": REVIEW_WEIGHTS.get(record["issue"], 0.10),
                }
            )

    exact_issue_rows = [
        row
        for row in sorted(
            score_rows,
            key=lambda r: (
                float(r["semantic_error_probability"]),
                float(r["review_priority_score"]),
                r["id"],
            ),
            reverse=True,
        )
        if row["has_lower_bound_semantic_issue"] == "yes"
    ]
    review_rows = [
        row
        for row in sorted(
            score_rows,
            key=lambda r: (
                float(r["review_priority_score"]),
                float(r["semantic_error_probability"]),
                r["has_lower_bound_semantic_issue"],
                r["id"],
            ),
            reverse=True,
        )
        if float(row["review_priority_score"]) >= 0.90 or row["has_lower_bound_semantic_issue"] == "yes"
    ]

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / f"{args.prefix}_scores.csv", score_rows)
    write_csv(args.outdir / f"{args.prefix}_issues.csv", issue_rows)
    write_csv(args.outdir / f"{args.prefix}_exact_issue_rows.csv", exact_issue_rows)
    write_csv(args.outdir / f"{args.prefix}_review_queue.csv", review_rows)
    summary = summarize(score_rows, issue_rows)
    summary["input_csv"] = str(args.input)
    summary["scores_csv"] = str(args.outdir / f"{args.prefix}_scores.csv")
    summary["issues_csv"] = str(args.outdir / f"{args.prefix}_issues.csv")
    summary["exact_issue_rows_csv"] = str(args.outdir / f"{args.prefix}_exact_issue_rows.csv")
    summary["review_queue_csv"] = str(args.outdir / f"{args.prefix}_review_queue.csv")
    (args.outdir / f"{args.prefix}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
