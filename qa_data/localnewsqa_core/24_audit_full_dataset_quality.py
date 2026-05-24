#!/usr/bin/env python3

import argparse
import csv
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
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
    "an",
    "and",
    "as",
    "at",
    "by",
    "de",
    "for",
    "from",
    "in",
    "la",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}

GENERIC_PAGE_TITLES = {
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

MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)

NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
}

ISSUE_METADATA = {
    "answer_not_in_options": "error",
    "ambiguous_answer_overlap": "warn",
    "ambiguous_same_answer": "error",
    "blacklisted_domain": "error",
    "duplicate_options": "error",
    "manual_reject": "error",
    "missing_certification": "error",
    "missing_expected_answer": "error",
    "option_count_not_4": "error",
    "explicit_missing_locale_in_question": "warn",
    "ambiguous_answer_leakage_in_question": "warn",
    "ambiguous_locale_leakage_in_question": "warn",
    "answer_absent_from_evidence_text": "warn",
    "boilerplate_excerpt": "warn",
    "generic_date_page_evidence": "warn",
    "generic_date_page_for_holiday_claim": "warn",
    "generic_number_page_evidence": "warn",
    "generic_page_without_context": "warn",
    "generic_title_no_locale": "warn",
    "locale_absent_from_evidence_text": "warn",
    "source_title_equals_url_or_blank": "warn",
}


def normalize(text: str) -> str:
    text = html.unescape(str(text or ""))
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    return [tok for tok in normalize(text).split() if tok and tok not in STOPWORDS]


def meaningful_tokens(text: str) -> set[str]:
    return {tok for tok in tokens(text) if len(tok) > 1}


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


def answer_in_options(answer: str, options: str) -> bool:
    answer_norm = normalize(answer)
    return bool(answer_norm) and answer_norm in {normalize(part) for part in option_parts(options)}


def answers_overlap(left: str, right: str) -> bool:
    left_tokens = tokens(left)
    right_tokens = tokens(right)
    if not left_tokens or not right_tokens:
        return False
    left_norm = " ".join(left_tokens)
    right_norm = " ".join(right_tokens)
    if left_norm == right_norm or left_norm in right_norm or right_norm in left_norm:
        return True
    return bool(meaningful_tokens(left) & meaningful_tokens(right))


def domain_of(url: str) -> str:
    return (urlparse(str(url or "")).netloc or "").lower()


def url_slug(url: str) -> str:
    path = unquote(urlparse(str(url or "")).path or "")
    stem = path.rsplit("/", 1)[-1].replace("_", " ")
    return normalize(stem)


def title_or_slug(row: dict, side: str) -> str:
    return normalize(row.get(f"{side}_evidence_title") or url_slug(row.get(f"{side}_evidence_url", "")))


def side_evidence_text(row: dict, side: str) -> str:
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
    text_norm = normalize(text)
    return any(normalize(marker) in text_norm for marker in COUNTRY_MARKERS.get(country, [country]))


def is_date_like(text: str) -> bool:
    text_norm = normalize(text)
    return bool(
        re.fullmatch(r"\d{1,2} " + "(" + "|".join(MONTHS) + r")", text_norm)
        or re.fullmatch("(" + "|".join(MONTHS) + r") \d{1,2}", text_norm)
        or re.fullmatch(r"\d{1,2} \d{1,2}", text_norm)
    )


def is_number_like(text: str) -> bool:
    text_norm = normalize(text)
    return bool(re.fullmatch(r"\d+", text_norm) or text_norm in NUMBER_WORDS)


def evidence_title_is_generic(title: str) -> bool:
    title_norm = normalize(title)
    return (
        title_norm in GENERIC_PAGE_TITLES
        or is_date_like(title_norm)
        or is_number_like(title_norm)
    )


def answer_mentions_question(answer: str, question: str) -> bool:
    answer_norm = normalize(answer)
    question_norm = normalize(question)
    if not answer_norm or not question_norm:
        return False
    if answer_norm in question_norm:
        return True
    answer_terms = meaningful_tokens(answer)
    if len(answer_terms) <= 1:
        return False
    return len(answer_terms & meaningful_tokens(question)) >= max(2, len(answer_terms) - 1)


def side_issues(row: dict, side: str) -> list[str]:
    issues = []
    answer = row.get(f"{side}_answer", "")
    country = row.get(f"{side}_country", "")
    url = row.get(f"{side}_evidence_url", "")
    title = row.get(f"{side}_evidence_title", "")
    excerpt = row.get(f"{side}_evidence_excerpt", "")
    snippet = row.get(f"{side}_evidence_snippet", "")
    cert = row.get(f"{side}_source_certified", "")
    evidence_text = side_evidence_text(row, side)
    evidence_title = title_or_slug(row, side)

    if not answer.strip():
        return issues
    if cert != "yes":
        issues.append("missing_certification")
    if url and domain_of(url) in BLACKLISTED_DOMAINS:
        issues.append("blacklisted_domain")
    if not title.strip() or title.strip() == url.strip():
        issues.append("source_title_equals_url_or_blank")
    if excerpt and "jump to content" in normalize(excerpt):
        issues.append("boilerplate_excerpt")

    answer_terms = meaningful_tokens(answer)
    if answer_terms and not answer_terms.issubset(set(evidence_text.split())):
        issues.append("answer_absent_from_evidence_text")
    if country and not has_country_marker(evidence_text, country):
        issues.append("locale_absent_from_evidence_text")

    if is_date_like(answer) and is_date_like(evidence_title):
        issues.append("generic_date_page_evidence")
        if row.get("topic") == "Public life and holidays":
            issues.append("generic_date_page_for_holiday_claim")
    if is_number_like(answer) and is_number_like(evidence_title):
        issues.append("generic_number_page_evidence")
    if evidence_title_is_generic(evidence_title) and country and not has_country_marker(evidence_text, country):
        issues.append("generic_title_no_locale")
    if evidence_title_is_generic(evidence_title) and not snippet.strip() and (
        not excerpt.strip() or "jump to content" in normalize(excerpt)
    ):
        issues.append("generic_page_without_context")
    return issues


def row_issues(row: dict) -> list[tuple[str, str]]:
    issues = []
    split = split_type(row)
    options = row.get("options", "")
    target_answer = row.get("target_answer", "")
    contrast_answer = row.get("contrast_answer", "")
    question = row.get("question", "")

    parts = option_parts(options)
    if len(parts) != 4:
        issues.append(("row", "option_count_not_4"))
    if len({normalize(part) for part in parts}) != len(parts):
        issues.append(("row", "duplicate_options"))
    if row.get("manual_review_status") == "manual_reject":
        issues.append(("row", "manual_reject"))
    if not target_answer.strip():
        issues.append(("target", "missing_expected_answer"))
    elif not answer_in_options(target_answer, options):
        issues.append(("target", "answer_not_in_options"))

    if split == "explicit":
        if row.get("target_country") and not has_country_marker(question, row.get("target_country", "")):
            issues.append(("target", "explicit_missing_locale_in_question"))
    else:
        if not contrast_answer.strip():
            issues.append(("contrast", "missing_expected_answer"))
        elif not answer_in_options(contrast_answer, options):
            issues.append(("contrast", "answer_not_in_options"))
        if normalize(target_answer) and normalize(target_answer) == normalize(contrast_answer):
            issues.append(("row", "ambiguous_same_answer"))
        elif answers_overlap(target_answer, contrast_answer):
            issues.append(("row", "ambiguous_answer_overlap"))
        if has_country_marker(question, row.get("target_country", "")) or has_country_marker(
            question, row.get("contrast_country", "")
        ):
            issues.append(("row", "ambiguous_locale_leakage_in_question"))
        if answer_mentions_question(target_answer, question) or answer_mentions_question(
            contrast_answer, question
        ):
            issues.append(("row", "ambiguous_answer_leakage_in_question"))

    for side in ("target", "contrast"):
        if side == "contrast" and split == "explicit":
            continue
        for issue in side_issues(row, side):
            issues.append((side, issue))
    return issues


def issue_record(row: dict, side: str, issue: str) -> dict:
    return {
        "id": row.get("id", ""),
        "split": split_type(row),
        "country": row.get("country", ""),
        "topic": row.get("topic", ""),
        "year": row.get("year", ""),
        "side": side,
        "severity": ISSUE_METADATA.get(issue, "warn"),
        "issue": issue,
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


def summarize(rows: list[dict], issue_rows: list[dict]) -> dict:
    issue_counts = Counter(record["issue"] for record in issue_rows)
    severity_counts = Counter(record["severity"] for record in issue_rows)
    row_issue_ids = {record["id"] for record in issue_rows}
    error_issue_ids = {record["id"] for record in issue_rows if record["severity"] == "error"}
    warn_issue_ids = {record["id"] for record in issue_rows if record["severity"] == "warn"}
    split_counts = Counter(split_type(row) for row in rows)
    topic_counts = Counter(row.get("topic", "") for row in rows)

    by_split_issue = defaultdict(Counter)
    by_topic_issue = defaultdict(Counter)
    for record in issue_rows:
        by_split_issue[record["split"]][record["issue"]] += 1
        by_topic_issue[record["topic"]][record["issue"]] += 1

    return {
        "rows": len(rows),
        "split_counts": dict(split_counts),
        "topic_counts": dict(topic_counts),
        "issue_occurrences": sum(issue_counts.values()),
        "rows_with_any_issue": len(row_issue_ids),
        "rows_with_error_issue": len(error_issue_ids),
        "rows_with_warn_issue": len(warn_issue_ids),
        "issue_counts": dict(issue_counts.most_common()),
        "severity_counts": dict(severity_counts),
        "by_split_issue_counts": {
            split: dict(counter.most_common()) for split, counter in sorted(by_split_issue.items())
        },
        "by_topic_issue_counts": {
            topic: dict(counter.most_common()) for topic, counter in sorted(by_topic_issue.items())
        },
        "issue_severity_map": ISSUE_METADATA,
        "notes": [
            "This is an internal heuristic audit, not a final human-validity estimate.",
            "Error issues are deterministic or already manually rejected; warn issues are review-risk flags.",
            "Generic date/number/title evidence flags capture cases where source matching may prove only the answer string, not the locale-specific claim.",
        ],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Heuristic quality audit for LocalNewsQA.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--prefix", default="full_35874_curated")
    args = parser.parse_args()

    with args.input.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    issue_rows = []
    for row in rows:
        for side, issue in row_issues(row):
            issue_rows.append(issue_record(row, side, issue))

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows, issue_rows)
    summary["input_csv"] = str(args.input)
    summary["issues_csv"] = str(args.outdir / f"{args.prefix}_issues.csv")
    summary["row_flags_csv"] = str(args.outdir / f"{args.prefix}_row_flags.csv")

    row_flags = {}
    for record in issue_rows:
        row_id = record["id"]
        if row_id not in row_flags:
            row_flags[row_id] = {
                key: record[key]
                for key in [
                    "id",
                    "split",
                    "country",
                    "topic",
                    "year",
                    "target_country",
                    "contrast_country",
                    "target_answer",
                    "contrast_answer",
                    "question",
                    "evidence_hint",
                    "manual_review_status",
                    "manual_review_reason",
                ]
            }
            row_flags[row_id]["issues"] = []
            row_flags[row_id]["severities"] = []
        row_flags[row_id]["issues"].append(f"{record['side']}:{record['issue']}")
        row_flags[row_id]["severities"].append(record["severity"])

    row_flag_rows = []
    for row_id, payload in row_flags.items():
        payload = dict(payload)
        payload["issues"] = " | ".join(payload["issues"])
        payload["max_severity"] = "error" if "error" in payload.pop("severities") else "warn"
        row_flag_rows.append(payload)

    write_csv(args.outdir / f"{args.prefix}_issues.csv", issue_rows)
    write_csv(args.outdir / f"{args.prefix}_row_flags.csv", row_flag_rows)
    (args.outdir / f"{args.prefix}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
