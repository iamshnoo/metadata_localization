#!/usr/bin/env python3

import argparse
import csv
import glob
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/semantic_quality_full_35874_scores.csv"
)
DEFAULT_OUTDIR = ROOT / "qa_data/localnewsqa_core/runs/quality_audit"
DEFAULT_CONTRAST_GLOB = str(
    ROOT
    / "results/downstream_localnewsqa_pretrained_figure9_contrast_full_multiseed"
    / "*"
    / "seed_*"
    / "localnewsqa_eval_contrast_without_metadata_custom_frozenfig9contrast_*_tminus_eminus_seed*_c0.jsonl"
)
DEFAULT_ANNOTATOR_SAMPLE = (
    ROOT
    / "latex/annotations/annotator_1/export/project-271-at-2026-05-13-05-16-9de32b09.csv"
)


STOPWORDS = {
    "a",
    "about",
    "after",
    "an",
    "and",
    "are",
    "article",
    "articles",
    "as",
    "at",
    "be",
    "before",
    "best",
    "by",
    "called",
    "centered",
    "common",
    "commonly",
    "countries",
    "country",
    "coverage",
    "did",
    "do",
    "does",
    "during",
    "fit",
    "fits",
    "for",
    "from",
    "had",
    "has",
    "have",
    "headline",
    "in",
    "into",
    "is",
    "it",
    "its",
    "likely",
    "local",
    "main",
    "major",
    "meant",
    "most",
    "name",
    "national",
    "new",
    "news",
    "of",
    "often",
    "old",
    "on",
    "or",
    "over",
    "referred",
    "referring",
    "report",
    "reports",
    "said",
    "says",
    "story",
    "than",
    "that",
    "the",
    "their",
    "them",
    "this",
    "to",
    "type",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whose",
    "why",
    "widely",
    "with",
    "would",
}

QUESTION_STARTERS = {
    "A",
    "An",
    "After",
    "Before",
    "Coverage",
    "During",
    "For",
    "From",
    "How",
    "In",
    "News",
    "On",
    "Reports",
    "The",
    "What",
    "When",
    "Where",
    "Which",
    "Who",
    "Why",
}

JOINER_WORDS = {"of", "and", "the", "for", "to", "de", "la", "le", "du", "van", "bin"}

COUNTRY_MARKERS = {
    "Bangladesh": ["Bangladesh", "Bangladeshi"],
    "Canada": ["Canada", "Canadian"],
    "Ghana": ["Ghana", "Ghanaian"],
    "Hong Kong": ["Hong Kong"],
    "India": ["India", "Indian"],
    "Ireland": ["Ireland", "Irish"],
    "Jamaica": ["Jamaica", "Jamaican"],
    "Kenya": ["Kenya", "Kenyan"],
    "Malaysia": ["Malaysia", "Malaysian"],
    "Nigeria": ["Nigeria", "Nigerian"],
    "Pakistan": ["Pakistan", "Pakistani"],
    "Philippines": ["Philippines", "Philippine", "Filipino"],
    "South Africa": ["South Africa", "South African"],
    "Sri Lanka": ["Sri Lanka", "Sri Lankan"],
    "Tanzania": ["Tanzania", "Tanzanian"],
    "United Kingdom": ["United Kingdom", "U.K.", "UK", "Britain", "British", "England", "English"],
    "United States": ["United States", "U.S.", "US", "USA", "American", "America"],
}

GENERIC_SPANS = {
    "agency",
    "authority",
    "bank",
    "cabinet",
    "city",
    "college",
    "commission",
    "conservative party",
    "court",
    "county",
    "department",
    "democratic party",
    "election commission",
    "federal",
    "football",
    "government",
    "high court",
    "house",
    "house of commons",
    "house of representatives",
    "labour party",
    "league",
    "ministry",
    "national",
    "office",
    "parliament",
    "police",
    "president",
    "prime minister",
    "public",
    "regulator",
    "regulatory authority",
    "republican party",
    "school",
    "senate",
    "speaker",
    "state",
    "stock exchange",
    "supreme court",
    "university",
}

TOPIC_PRIOR = {
    "Sports": 0.7,
    "Public life and holidays": 0.6,
    "Geography and infrastructure": 0.5,
    "Media and culture": 0.4,
    "Education": 0.3,
    "Politics and government": 0.2,
    "Economy and business": 0.2,
    "Civic institutions": 0.1,
}


def normalize(text: str) -> str:
    text = html.unescape(str(text or ""))
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9+]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    return [tok for tok in normalize(text).split() if len(tok) > 1 and tok not in STOPWORDS]


def token_set(text: str) -> set[str]:
    return set(tokens(text))


def contains_phrase(text: str, phrase: str) -> bool:
    phrase_norm = normalize(phrase)
    return bool(phrase_norm) and phrase_norm in normalize(text)


def option_parts(options: str) -> list[str]:
    parts = [part.strip() for part in str(options or "").split("||") if part.strip()]
    cleaned = []
    for part in parts:
        if re.match(r"^[A-D]:", part):
            part = part.split(":", 1)[1].strip()
        cleaned.append(part)
    return cleaned


def contrast_key(row: dict) -> tuple[str, str, str, str]:
    return (
        row.get("country", ""),
        row.get("question", ""),
        row.get("target_answer") or row.get("correct_answer") or "",
        row.get("contrast_answer") or "",
    )


def is_capitalish(word: str) -> bool:
    clean = word.strip("'\".,:;!?()[]{}")
    return bool(
        re.match(r"^(?:[A-Z][A-Za-z0-9'.+-]*|[A-Z]{2,}|[0-9]+[A-Za-z]*|[0-9]+\+[0-9]+)$", clean)
    )


def extract_named_spans(question: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'.+-]*", question)
    spans = []
    current = []
    for word in words:
        clean = word.strip("'\".,:;!?()[]{}")
        if clean in QUESTION_STARTERS:
            if current:
                spans.append(" ".join(current))
                current = []
            continue

        if is_capitalish(clean):
            current.append(clean)
        elif current and clean.lower() in JOINER_WORDS:
            current.append(clean)
        else:
            if current:
                spans.append(" ".join(current))
                current = []
    if current:
        spans.append(" ".join(current))

    filtered = []
    for span in spans:
        span = re.sub(r"\s+", " ", span).strip()
        span_norm = normalize(span)
        if not span_norm or len(span_norm) < 3:
            continue
        if span_norm in GENERIC_SPANS:
            continue
        parts = span_norm.split()
        raw_no_punct = re.sub(r"[^A-Za-z0-9]", "", span)
        is_acronym = bool(re.match(r"^[A-Z0-9]{2,}$", raw_no_punct))
        if len(parts) == 1 and not is_acronym and parts[0] in GENERIC_SPANS:
            continue
        filtered.append(span)
    return list(dict.fromkeys(filtered))


def add_flag(flags: list[str], flag: str, score_parts: Counter, points: float) -> None:
    flags.append(flag)
    score_parts[flag] += points


def score_rule_based(row: dict) -> tuple[float, list[str], list[str], list[str]]:
    question = row["question"]
    question_norm = normalize(question)
    question_tokens = token_set(question)
    target_answer_tokens = token_set(row.get("target_answer", ""))
    contrast_answer_tokens = token_set(row.get("contrast_answer", ""))
    all_answer_tokens = target_answer_tokens | contrast_answer_tokens
    target_blob = " ".join(
        [
            row.get("target_evidence_title", ""),
            row.get("target_evidence_snippet", ""),
            row.get("target_evidence_excerpt", ""),
            row.get("target_answer", ""),
        ]
    )
    contrast_blob = " ".join(
        [
            row.get("contrast_evidence_title", ""),
            row.get("contrast_evidence_snippet", ""),
            row.get("contrast_evidence_excerpt", ""),
            row.get("contrast_answer", ""),
        ]
    )

    flags = []
    anchors = []
    score_parts = Counter()

    for country in {row.get("country", ""), row.get("target_country", "")}:
        for marker in COUNTRY_MARKERS.get(country, []):
            if contains_phrase(question, marker):
                add_flag(flags, "target_country_marker_in_question", score_parts, 3.0)
                anchors.append(marker)
                break

    for marker in COUNTRY_MARKERS.get(row.get("contrast_country", ""), []):
        if contains_phrase(question, marker):
            add_flag(flags, "contrast_country_marker_in_question", score_parts, 1.0)
            anchors.append(marker)
            break

    target_answer_hits = question_tokens & target_answer_tokens
    contrast_answer_hits = question_tokens & contrast_answer_tokens
    if target_answer_hits and not contrast_answer_hits:
        add_flag(flags, "target_answer_token_in_question", score_parts, 2.0)
        anchors.extend(sorted(target_answer_hits))
    if len(target_answer_hits) >= 2:
        add_flag(flags, "multiple_target_answer_tokens_in_question", score_parts, 1.0)

    named_spans = extract_named_spans(question)
    non_answer_spans = []
    target_only_spans = []
    for span in named_spans:
        span_tokens = token_set(span)
        if not span_tokens:
            continue
        if not (span_tokens & all_answer_tokens):
            non_answer_spans.append(span)
        if contains_phrase(target_blob, span) and not contains_phrase(contrast_blob, span):
            target_only_spans.append(span)

    if non_answer_spans:
        add_flag(
            flags,
            "non_answer_named_anchor_in_question",
            score_parts,
            min(2.6, 1.4 + 0.4 * (len(non_answer_spans) - 1)),
        )
        anchors.extend(non_answer_spans[:6])

    if target_only_spans:
        add_flag(
            flags,
            "target_evidence_only_anchor",
            score_parts,
            min(2.5, 1.5 + 0.3 * (len(target_only_spans) - 1)),
        )
        anchors.extend(target_only_spans[:6])

    pattern_rules = [
        (
            "named_person_action_frame",
            r"\bwhich (?:team|party|office|chamber|company|body) did\b|\bwhere .+ served\b|\bwho .+ served\b",
            2.0,
        ),
        (
            "specific_headline_event_frame",
            r"\b(?:headline|story|coverage|article|articles|report|reports|roundup)\b.*\b(?:opening|collapse|assault|pardon|rebuilding|championship|controversy|record high|camp|preservation grants|landmark listing|beat estimates)\b",
            2.0,
        ),
        (
            "named_venue_landmark_frame",
            r"\b(?:bridge|terminal|theater|theatre|canyon|borough|city region|state by population|site of the collapse|hosted|venue)\b",
            1.8,
        ),
        (
            "holiday_ritual_frame",
            r"\b(?:turkey pardon|veterans?|cemetery|indianapolis|presidents day|thanksgiving|memorial day|bank holiday|remembrance)\b",
            2.0,
        ),
        (
            "exam_policy_specificity_frame",
            r"\b(?:10\+2|answer keys|cutoffs|advanced paper|single national undergraduate|premier engineering institute|entrance exam)\b",
            2.0,
        ),
    ]
    for flag, pattern, points in pattern_rules:
        if re.search(pattern, question_norm):
            add_flag(flags, flag, score_parts, points)

    if re.search(r"['\"][a-z][^'\"]{2,}['\"]", question) and re.search(
        r"\bwhat is (?:a|an|the)\b", question_norm
    ):
        add_flag(flags, "generic_quoted_term_definition", score_parts, 2.0)

    if row.get("topic") == "Public life and holidays" and re.search(
        r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
        question_norm,
    ):
        add_flag(flags, "holiday_month_anchor", score_parts, 1.0)

    extra_years = [
        year
        for year in re.findall(r"\b(?:19|20)\d{2}\b", question)
        if year != str(row.get("year", ""))
    ]
    if extra_years:
        add_flag(flags, "extra_year_in_question", score_parts, 0.8)
        anchors.extend(extra_years)

    score = sum(score_parts.values())
    if score > 0:
        score += TOPIC_PRIOR.get(row.get("topic", ""), 0.0)

    return score, sorted(set(flags)), list(dict.fromkeys(anchors)), named_spans


def load_contrast_model_votes(paths: list[str]) -> dict[tuple[str, str, str, str], Counter]:
    votes: dict[tuple[str, str, str, str], Counter] = defaultdict(Counter)
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("split_type") != "ambiguous" and not row.get("ambiguity_flag"):
                    continue
                answer = row.get("processed_answer") or ""
                target = row.get("target_answer") or row.get("correct_answer") or ""
                contrast = row.get("contrast_answer") or ""
                if answer == target:
                    label = "target"
                elif answer == contrast:
                    label = "contrast"
                else:
                    label = "other"
                votes[contrast_key(row)][label] += 1
    return votes


def risk_tier(rule_score: float, target_share: float, final_score: float, flags: list[str]) -> str:
    if (
        "target_country_marker_in_question" in flags
        or rule_score >= 3.0
        or target_share >= 0.80
        or final_score >= 4.5
    ):
        return "high"
    if rule_score >= 2.0 or target_share >= 0.50 or final_score >= 3.0:
        return "medium"
    if flags or target_share >= 0.30:
        return "low"
    return "clean"


def annotate_rows(rows: list[dict], model_votes: dict[tuple[str, str, str, str], Counter]) -> list[dict]:
    out = []
    for row in rows:
        rule_score, flags, anchors, spans = score_rule_based(row)
        votes = model_votes.get(contrast_key(row), Counter())
        vote_n = sum(votes.values())
        target_share = votes["target"] / vote_n if vote_n else 0.0
        contrast_share = votes["contrast"] / vote_n if vote_n else 0.0
        other_share = votes["other"] / vote_n if vote_n else 0.0
        final_score = rule_score + 2.0 * target_share
        if target_share >= 0.70 and contrast_share < 0.20:
            final_score += 0.8
        output_flags = list(flags)
        if target_share >= 0.80:
            output_flags.append("model_strong_target_answer_bias_without_metadata")
        elif target_share >= 0.50:
            output_flags.append("model_moderate_target_answer_bias_without_metadata")
        tier = risk_tier(rule_score, target_share, final_score, output_flags)

        out_row = dict(row)
        out_row.update(
            {
                "weak_locale_risk": tier,
                "weak_locale_score": f"{final_score:.3f}",
                "rule_score": f"{rule_score:.3f}",
                "model_vote_n": str(vote_n),
                "model_target_answer_share_without_metadata": f"{target_share:.3f}",
                "model_contrast_answer_share_without_metadata": f"{contrast_share:.3f}",
                "model_other_answer_share_without_metadata": f"{other_share:.3f}",
                "weak_locale_flags": " | ".join(sorted(set(output_flags))),
                "weak_locale_anchors": " | ".join(anchors),
                "named_spans": " | ".join(spans),
            }
        )
        out.append(out_row)
    return out


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(scored: list[dict], annotator_sample: Path | None = None) -> dict:
    total = len(scored)
    risk_counts = Counter(row["weak_locale_risk"] for row in scored)
    by_topic = defaultdict(Counter)
    by_country = defaultdict(Counter)
    by_pair = defaultdict(Counter)
    by_flag = Counter()
    for row in scored:
        risk = row["weak_locale_risk"]
        by_topic[row["topic"]][risk] += 1
        by_country[row["country"]][risk] += 1
        by_pair[f"{row['country']} -> {row['contrast_country']}"][risk] += 1
        for flag in row["weak_locale_flags"].split(" | "):
            if flag:
                by_flag[flag] += 1

    summary = {
        "total_ambiguous_rows": total,
        "risk_counts": dict(risk_counts),
        "risk_rates": {risk: risk_counts[risk] / total for risk in sorted(risk_counts)},
        "high_or_medium_count": risk_counts["high"] + risk_counts["medium"],
        "high_or_medium_rate": (risk_counts["high"] + risk_counts["medium"]) / total if total else 0.0,
        "by_topic": {k: dict(v) for k, v in sorted(by_topic.items())},
        "by_country": {k: dict(v) for k, v in sorted(by_country.items())},
        "by_country_pair": {k: dict(v) for k, v in sorted(by_pair.items())},
        "flag_counts": dict(by_flag.most_common()),
    }

    if annotator_sample and annotator_sample.exists():
        by_question = {row["question"]: row for row in scored}
        eval_counts = defaultdict(Counter)
        with open(annotator_sample, newline="", encoding="utf-8") as handle:
            for ann in csv.DictReader(handle):
                if ann.get("split") != "ambiguous" or not ann.get("human_reject_reason"):
                    continue
                scored_row = by_question.get(ann["question"])
                if not scored_row:
                    continue
                label = "reject" if ann["human_reject_reason"] != "accept" else "accept"
                risk = scored_row["weak_locale_risk"]
                eval_counts[label][risk] += 1
        summary["annotator_1_sample_calibration"] = {
            label: dict(counts) for label, counts in sorted(eval_counts.items())
        }

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flag ambiguous LocalNewsQA rows whose question text appears to force the target locale."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--contrast-eval-glob",
        default=DEFAULT_CONTRAST_GLOB,
        help="Glob for contrast/no-metadata eval jsonl files used as model-vote leakage signal.",
    )
    parser.add_argument("--annotator-sample", type=Path, default=DEFAULT_ANNOTATOR_SAMPLE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.input, newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == "ambiguous"]

    contrast_paths = sorted(glob.glob(args.contrast_eval_glob))
    model_votes = load_contrast_model_votes(contrast_paths) if contrast_paths else {}
    scored = annotate_rows(rows, model_votes)
    scored.sort(key=lambda row: (-float(row["weak_locale_score"]), row["id"]))

    output_fields = [
        "id",
        "weak_locale_risk",
        "weak_locale_score",
        "rule_score",
        "model_vote_n",
        "model_target_answer_share_without_metadata",
        "model_contrast_answer_share_without_metadata",
        "model_other_answer_share_without_metadata",
        "weak_locale_flags",
        "weak_locale_anchors",
        "country",
        "contrast_country",
        "topic",
        "year",
        "question",
        "target_answer",
        "contrast_answer",
        "options",
        "evidence_hint",
        "target_evidence_title",
        "contrast_evidence_title",
        "target_evidence_url",
        "contrast_evidence_url",
        "semantic_risk_band",
        "review_priority_band",
        "issues",
    ]
    full_path = args.outdir / "weak_locale_ambiguous_flags.csv"
    write_csv(full_path, scored, output_fields)

    review_rows = [row for row in scored if row["weak_locale_risk"] in {"high", "medium"}]
    review_path = args.outdir / "weak_locale_ambiguous_review_queue.csv"
    write_csv(review_path, review_rows, output_fields)

    summary = summarize(scored, args.annotator_sample)
    summary.update(
        {
            "input": str(args.input),
            "contrast_eval_files": contrast_paths,
            "full_flags_csv": str(full_path),
            "review_queue_csv": str(review_path),
            "method": {
                "rule_signal": "question-text target/contrast asymmetry: country markers, target-answer tokens, named anchors, event/holiday/exam/landmark frames",
                "model_signal": "pretrained contrast-eval no-metadata runs; high target-answer share means the question tends to force the target answer even when scored for contrast",
                "risk_tiers": {
                    "high": "clear target locale/event leakage, strong rule score, or >=80% target-answer model vote",
                    "medium": "moderate rule signal or >=50% target-answer model vote",
                    "low": "some weak signal",
                    "clean": "no detector signal",
                },
            },
        }
    )
    summary_path = args.outdir / "weak_locale_ambiguous_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
