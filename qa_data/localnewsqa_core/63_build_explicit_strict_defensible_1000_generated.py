#!/usr/bin/env python3

import csv
import importlib.util
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

import requests


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_semantic_gold_ambiguous_1700"
AUDIT_DIR = BASE / "explicit_max_audit"
SOURCE_STRICT = AUDIT_DIR / "strict_defensible_954/localnewsqa_targetqa_explicit_strict_defensible_954_per_country.jsonl"
AMBIGUOUS = BASE / "localnewsqa_ambiguous_semantic_gold_1700.jsonl"
OUTDIR = AUDIT_DIR / "strict_defensible_1000_generated"
OUT_JSONL = OUTDIR / "localnewsqa_targetqa_explicit_strict_defensible_1000_per_country_generated.jsonl"
OUT_CSV = OUTDIR / "localnewsqa_targetqa_explicit_strict_defensible_1000_per_country_generated.csv"
SUMMARY = OUTDIR / "strict_defensible_1000_generated_build_summary.json"
GEN_LOG = OUTDIR / "generated_question_log.csv"
REJECT_LOG = OUTDIR / "generated_question_reject_log.csv"
FETCH_CACHE = OUTDIR / "strict_defensible_1000_target_evidence_fetches.jsonl"

CANDIDATE_POOLS = [
    AUDIT_DIR / "strict_1000/localnewsqa_targetqa_explicit_strict_1000_per_country_polished.jsonl",
    AUDIT_DIR / "localnewsqa_targetqa_explicit_style_max_strict_no_warnings.jsonl",
    AUDIT_DIR / "localnewsqa_targetqa_explicit_style_max_paper_clean.jsonl",
    AUDIT_DIR / "localnewsqa_targetqa_explicit_style_max_clean.jsonl",
]
FETCH_CACHES = [
    BASE / "semantic_gold_selected_evidence_fetches.jsonl",
    AUDIT_DIR / "explicit_target_evidence_fetches.jsonl",
    AUDIT_DIR / "strict_1000/polished_audit/explicit_target_evidence_fetches.jsonl",
    AUDIT_DIR / "strict_defensible_954/strict_defensible_target_evidence_fetches.jsonl",
    AUDIT_DIR / "strict_defensible_954/audit/explicit_target_evidence_fetches.jsonl",
    FETCH_CACHE,
]

AUDIT_EXPLICIT_SCRIPT = ROOT / "qa_data/localnewsqa_core/48_audit_explicit_max_split.py"
FETCH_AUDIT_SCRIPT = ROOT / "qa_data/localnewsqa_core/32_web_audit_ambiguous_verifiable.py"
BUILDER_SCRIPT = ROOT / "qa_data/localnewsqa_core/45_build_relation_strict_gold_ambiguous.py"

COUNTRY_ORDER = [
    "Bangladesh",
    "Canada",
    "Ghana",
    "Hong Kong",
    "India",
    "Ireland",
    "Jamaica",
    "Kenya",
    "Malaysia",
    "Nigeria",
    "Pakistan",
    "Philippines",
    "South Africa",
    "Sri Lanka",
    "Tanzania",
    "United Kingdom",
    "United States",
]

SOURCE_PRIORITY = {
    "explicit": 0,
    "ambiguous_salvaged_target": 1,
    "ambiguous_pool_salvaged_target_web_supported": 2,
    "explicit_generated_strict": 3,
}

BAD_LEAD_FRAGMENTS = {
    "jump to content",
    "main menu",
    "toggle the table",
    "edit links",
    "article talk",
    "from wikipedia",
    "wiktionary",
    "may refer to",
    "can refer to",
    "commonly refers to",
    "topics referred to by the same term",
    "find sources",
    "website ",
    "products ",
    "company type",
    "traded as",
    "isin ",
    "jump to navigation",
}

BAD_GENERATED_QUESTION_FRAGMENTS = {
    " associated with ",
    " the answer",
    " answer ",
    " find sources",
    " website ",
    " products ",
    " company type ",
    " traded as ",
    " toggle ",
    " edit ",
    " may refer ",
    " topics referred ",
    " jump to ",
    " see also ",
    " http ",
    " https ",
    " _( ",
    " it is ",
    " it was ",
    " it has ",
}

LANGUAGE_LABELS = (
    "Arabic",
    "Bengali",
    "Chinese",
    "Hindi",
    "Malay",
    "Sinhala",
    "Swahili",
    "Tamil",
    "Urdu",
)

COUNTRY_MARKERS = {
    "Bangladesh": ["bangladesh", "bangladeshi"],
    "Canada": ["canada", "canadian"],
    "Ghana": ["ghana", "ghanaian"],
    "Hong Kong": ["hong kong", "hongkonger", "hong konger"],
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
    "United Kingdom": ["united kingdom", "uk", "british", "england", "scotland", "wales"],
    "United States": ["united states", "u s", "us", "american"],
}


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


def write_fetch_cache(path: Path, fetches: dict[str, dict], selected_urls: set[str]) -> None:
    write_jsonl(path, [fetches[url] for url in sorted(selected_urls) if url in fetches])


def norm(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def rank(row: dict) -> tuple:
    return (
        SOURCE_PRIORITY.get(row.get("source_split_type", ""), 9),
        norm(row.get("target_answer", "")),
        norm(row.get("question", "")),
        row.get("source_row_id", ""),
    )


def wiki_title_from_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc not in {"en.wikipedia.org", "en.m.wikipedia.org"} or not parsed.path.startswith("/wiki/"):
        return ""
    return unquote(parsed.path[len("/wiki/") :]).replace("_", " ").strip()


def fetch_wikipedia_extracts(urls: list[str]) -> dict[str, str]:
    pairs = [(url, wiki_title_from_url(url)) for url in urls if wiki_title_from_url(url)]
    out: dict[str, str] = {}
    if not pairs:
        return out
    api_url = "https://en.wikipedia.org/w/api.php"
    for start in range(0, len(pairs), 50):
        batch = pairs[start : start + 50]
        try:
            response = requests.get(
                api_url,
                params={
                    "action": "query",
                    "format": "json",
                    "prop": "extracts|info",
                    "explaintext": 1,
                    "exintro": 1,
                    "exsentences": 3,
                    "redirects": 1,
                    "inprop": "url",
                    "titles": "|".join(title for _, title in batch),
                },
                timeout=10,
                headers={"User-Agent": "LocalNewsQA-strict-explicit-generation/1.0"},
            )
            response.raise_for_status()
            data = response.json().get("query", {})
        except Exception:
            continue
        redirects = {item.get("from", ""): item.get("to", "") for item in data.get("redirects", [])}
        normalized = {item.get("from", ""): item.get("to", "") for item in data.get("normalized", [])}
        pages = {page.get("title", ""): page for page in data.get("pages", {}).values()}
        for url, title in batch:
            final_title = redirects.get(title, normalized.get(title, title))
            final_title = redirects.get(final_title, normalized.get(final_title, final_title))
            page = pages.get(final_title) or pages.get(title)
            text = re.sub(r"\s+", " ", str(page.get("extract", "") if page else "")).strip()
            if text:
                out[url] = text
        time.sleep(0.05)
    return out


def fetch_wikipedia_summary(url: str) -> str:
    title = wiki_title_from_url(url)
    if not title:
        return ""
    api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title, safe='')}"
    try:
        response = requests.get(
            api_url,
            timeout=12,
            headers={"User-Agent": "LocalNewsQA-strict-explicit-generation/1.0"},
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return ""
    if data.get("type") == "disambiguation":
        return ""
    return re.sub(r"\s+", " ", str(data.get("extract", "") or "")).strip()


def prepare_text(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]", "", str(text or ""))
    marker = "From Wikipedia, the free encyclopedia"
    if marker in text:
        text = text.split(marker, 1)[1]
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str) -> list[str]:
    text = prepare_text(text)
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def clean_sentence(sentence: str) -> str:
    sentence = re.sub(r"\([^)]{0,100}\)", "", sentence)
    for label in LANGUAGE_LABELS:
        sentence = re.sub(rf"\b{label}:\s*[^,;()]+,?\s*", "", sentence)
    sentence = re.sub(r"\s+[A-Z][a-z]+ languages?\b.*$", "", sentence)
    sentence = re.sub(r"\s+Edit links?\b.*$", "", sentence)
    sentence = re.sub(r"\s+Toggle .*?$", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence.strip(" .")


def answer_in_text(text: str, row: dict, fetch_audit: Any) -> bool:
    return fetch_audit.contains_any(fetch_audit.normalize_text(text), fetch_audit.answer_aliases(row.get("target_answer", "")))


def first_good_sentence(row: dict, fetches: dict[str, dict], wiki_extracts: dict[str, str], fetch_audit: Any) -> str:
    url = row.get("target_evidence_url", "")
    texts = [
        row.get("target_evidence_excerpt", ""),
        wiki_extracts.get(url, ""),
        fetches.get(url, {}).get("text", ""),
    ]
    for text in texts:
        for sentence in split_sentences(str(text or "")[:6000])[:12]:
            sentence = clean_sentence(sentence)
            sentence_norm = norm(sentence)
            if any(fragment in sentence_norm for fragment in BAD_LEAD_FRAGMENTS):
                continue
            if len(sentence) < 25 or len(sentence) > 300:
                continue
            if answer_in_text(sentence, row, fetch_audit):
                return sentence
        span = extract_answer_predicate_span(str(text or ""), row, fetch_audit)
        if span:
            return span
    # Fall back only if metadata itself is answer-bearing and non-generic.
    meta = clean_sentence(row.get("target_evidence_excerpt", ""))
    if len(meta) >= 35 and answer_in_text(meta, row, fetch_audit):
        return meta
    return ""


def legal_name_suffix_pattern() -> str:
    suffixes = [
        "Limited",
        "Ltd\\.?",
        "PLC",
        "plc",
        "Inc\\.?",
        "Corporation",
        "Corp\\.?",
        "Company",
        "Bank",
        "University",
        "Stadium",
        "F\\.?C\\.?",
        "FC",
    ]
    return rf"(?:\s+(?:{'|'.join(suffixes)}))?"


def answer_patterns(row: dict, fetch_audit: Any, include_titles: bool = False) -> list[str]:
    aliases = set(fetch_audit.answer_aliases(row.get("target_answer", "")))
    aliases.add(str(row.get("target_answer", "") or "").strip())
    if include_titles:
        aliases.update(
            str(value or "").strip()
            for value in [
                row.get("target_evidence_title", ""),
                wiki_title_from_url(row.get("target_evidence_url", "")),
            ]
        )
    aliases = {alias for alias in aliases if alias}
    return [re.escape(alias) for alias in sorted(aliases, key=len, reverse=True)]


def extract_answer_predicate_span(text: str, row: dict, fetch_audit: Any) -> str:
    prepared = prepare_text(text)
    if not prepared:
        return ""
    alias_pattern = "|".join(answer_patterns(row, fetch_audit))
    if not alias_pattern:
        return ""
    suffix = legal_name_suffix_pattern()
    match = re.search(
        rf"\b(?:the\s+)?(?:{alias_pattern}){suffix}(?:\s*,\s*[^,]{{2,90}}\s*,)?\s+"
        rf"(?:is|was|are|were|has|had|serves|served|operates|operated|became|"
        rf"becomes|remains|opened|founded|established|located)\b[^.!?]{{12,220}}",
        prepared,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    span = clean_sentence(match.group(0))
    span_norm = norm(span)
    if any(fragment in span_norm for fragment in BAD_LEAD_FRAGMENTS):
        return ""
    if len(span) < 30 or len(span) > 280:
        return ""
    return span


def predicate_after_answer(sentence: str, row: dict, fetch_audit: Any) -> str:
    cleaned = clean_sentence(sentence)
    alias_pattern = "|".join(answer_patterns(row, fetch_audit))
    if not alias_pattern:
        return ""
    suffix = legal_name_suffix_pattern()

    leading = re.match(
        rf"^(?:the\s+)?(?:{alias_pattern}){suffix}(?:\s*,\s*[^,]{{2,90}}\s*,)?\s+"
        rf"(?P<predicate>(?:is|was|are|were|has|had|serves|served|operates|operated|became|"
        rf"becomes|remains|opened|founded|established|located)\b.+)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if leading:
        return leading.group("predicate").strip(" .")

    known_as = re.search(
        rf"\b(?:known as|known professionally as|known by (?:his|her|their) stage name|popularly known as)\s+"
        rf"(?:{alias_pattern})\s*,?\s+(?P<predicate>(?:is|was|are|were)\b.+)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if known_as:
        return known_as.group("predicate").strip(" .")

    return ""


def strip_answer(sentence: str, row: dict, fetch_audit: Any) -> str:
    out = clean_sentence(sentence)
    aliases = sorted(fetch_audit.answer_aliases(row.get("target_answer", "")), key=len, reverse=True)
    aliases.extend([row.get("target_evidence_title", ""), row.get("target_answer", "")])
    seen = set()
    for alias in aliases:
        alias = str(alias or "").strip()
        if not alias or alias.lower() in seen:
            continue
        seen.add(alias.lower())
        out = re.sub(rf"\b{re.escape(alias)}\b", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\s+([,.;:])", r"\1", out)
    out = re.sub(r"^the\s*,?\s*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"^,?\s*(?:officially [^,]+,\s*)?(?:is|was|are|were)\s+", "", out, flags=re.IGNORECASE)
    out = re.sub(r",\s*(?:is|was|are|were)\s+", ", ", out, flags=re.IGNORECASE)
    out = re.sub(r"\bknown by (?:his|her|their) stage name\s*,", "known by that stage name,", out, flags=re.IGNORECASE)
    out = re.sub(r"\bknown as\s*,", "known by that name,", out, flags=re.IGNORECASE)
    out = re.sub(r"\bpopularly known as\s*,", "popularly known by that name,", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\s+", " ", out).strip(" .")
    return out


def infer_entity_type(row: dict, description: str) -> str:
    answer_blob = norm(" ".join([row.get("target_evidence_title", ""), row.get("target_answer", "")]))
    full_blob = norm(" ".join([row.get("target_evidence_title", ""), row.get("target_answer", ""), description]))
    answer_padded = f" {answer_blob} "
    full_padded = f" {full_blob} "
    answer_title_checks = [
        ("stock exchange", [" stock exchange "]),
        ("stadium", [" stadium ", " arena "]),
        ("airport", [" airport "]),
        ("port", [" port "]),
        ("university", [" university ", " college "]),
        ("school", [" school "]),
        ("government body", [" ministry ", " department ", " bureau ", " authority ", " commission ", " agency ", " council ", " office "]),
        ("company", [" company ", " corporation ", " bank ", " retailer ", " airline ", " operator ", " conglomerate ", " telecom ", " manufacturer ", " group "]),
        ("newspaper", [" newspaper ", " daily "]),
        ("broadcaster", [" broadcaster ", " television network ", " tv channel ", " radio ", " channel "]),
        ("television series", [" television series "]),
        ("football club", [" football club ", " fc "]),
    ]
    for label, needles in answer_title_checks:
        if any(needle in answer_padded for needle in needles):
            return label
    full_checks = [
        (
            "person",
            [
                " politician ",
                " singer ",
                " actor ",
                " actress ",
                " footballer ",
                " cricketer ",
                " writer ",
                " artist ",
                " minister ",
                " president ",
                " musician ",
                " rapper ",
                " filmmaker ",
                " director ",
                " producer ",
                " media personality ",
                " entrepreneur ",
                " table tennis player ",
                " athlete ",
            ],
        ),
        ("television series", [" television series ", " animated television series "]),
        ("newspaper", [" newspaper ", " daily "]),
        ("broadcaster", [" broadcaster ", " television network ", " tv channel ", " radio ", " channel "]),
        ("company", [" company ", " corporation ", " bank ", " retailer ", " airline ", " operator ", " conglomerate ", " telecom ", " manufacturer ", " group "]),
        ("government body", [" ministry ", " department ", " bureau ", " authority ", " commission ", " agency ", " council ", " office "]),
        ("political party", [" political party ", " party "]),
        ("competition", [" competition ", " tournament ", " league ", " cup ", " championship ", " festival "]),
        ("qualification", [" examination ", " certificate ", " qualification "]),
        ("holiday", [" holiday ", " celebration "]),
        ("university", [" university ", " college "]),
        ("school", [" school "]),
        ("city", [" city ", " capital ", " metropolitan "]),
        ("town", [" town ", " municipality "]),
        ("district", [" district "]),
        ("region", [" region ", " province ", " state "]),
        ("date", [" date ", " day "]),
    ]
    for label, needles in full_checks:
        if any(needle in full_padded for needle in needles):
            return label
    return "answer"


def generate_question(row: dict, fetches: dict[str, dict], wiki_extracts: dict[str, str], fetch_audit: Any) -> tuple[str, str]:
    sentence = first_good_sentence(row, fetches, wiki_extracts, fetch_audit)
    if not sentence:
        return "", "no_good_sentence"
    sentence_norm = norm(sentence)
    country_markers = COUNTRY_MARKERS.get(row.get("country", ""), [norm(row.get("country", ""))])
    if country_markers and not any(marker in sentence_norm for marker in country_markers):
        return "", "support_missing_country_marker"
    predicate = predicate_after_answer(sentence, row, fetch_audit)
    if not predicate:
        return "", "no_subject_predicate"
    if not predicate or len(predicate) < 12:
        return "", "description_too_short"
    if norm(predicate).startswith(("established ", "founded ", "located ")):
        return "", "bad_predicate_fragment"
    if any(fragment in norm(predicate) for fragment in BAD_LEAD_FRAGMENTS):
        return "", "bad_description_fragment"
    if len(predicate) > 180:
        predicate = predicate[:177].rsplit(" ", 1)[0]
    if answer_in_text(predicate, row, fetch_audit):
        return "", "answer_leak_after_strip"
    entity_type = infer_entity_type(row, predicate)
    if entity_type == "answer":
        return "", "generic_entity_type"
    question = f"In {row.get('country')}, which {entity_type} {predicate[0].lower() + predicate[1:]}?"
    question_norm = f" {norm(question)} "
    if any(fragment in question_norm for fragment in BAD_GENERATED_QUESTION_FRAGMENTS):
        return "", "bad_generated_question_fragment"
    if answer_in_text(question, row, fetch_audit):
        return "", "answer_leak_question"
    if len(question) > 230:
        return "", "question_too_long"
    return question, sentence


def high_conf_issue(row: dict, fetches: dict[str, dict], builder: Any) -> bool:
    text = builder.evidence_cue_text(row, "target", fetches)
    return any(cue["norm"] not in text for cue in builder.high_confidence_question_cues(row))


def anchor_issue(row: dict, fetches: dict[str, dict], fetch_audit: Any) -> bool:
    fetch = fetches.get(row.get("target_evidence_url", ""), {})
    anchor_text = " ".join(
        str(part or "")
        for part in [
            row.get("target_evidence_title", ""),
            row.get("target_evidence_excerpt", ""),
            fetch.get("title", ""),
            str(fetch.get("text", ""))[:1500],
        ]
    )
    return not fetch_audit.contains_any(fetch_audit.normalize_text(anchor_text), fetch_audit.answer_aliases(row.get("target_answer", "")))


def main() -> None:
    explicit_audit = load_module(AUDIT_EXPLICIT_SCRIPT, "audit48")
    fetch_audit = load_module(FETCH_AUDIT_SCRIPT, "fetch32")
    builder = load_module(BUILDER_SCRIPT, "builder45")
    fetches: dict[str, dict] = {}
    for cache in FETCH_CACHES:
        fetches.update(explicit_audit.load_fetch_cache(cache))

    ambiguous_ids = {row["source_row_id"] for row in read_jsonl(AMBIGUOUS)}
    selected = read_jsonl(SOURCE_STRICT)
    used_sources = {row["source_row_id"] for row in selected}
    used_questions = {norm(row["question"]) for row in selected}
    counts = Counter(row["country"] for row in selected)

    pool_rows = []
    seen_pool = set()
    for pool in CANDIDATE_POOLS:
        for row in read_jsonl(pool):
            source_id = row.get("source_row_id", "")
            if source_id in seen_pool:
                continue
            seen_pool.add(source_id)
            if source_id in ambiguous_ids:
                continue
            if row.get("country") not in COUNTRY_ORDER:
                continue
            pool_rows.append(dict(row))

    strict_existing_by_country: dict[str, list[dict]] = defaultdict(list)
    generation_source_by_country: dict[str, list[dict]] = defaultdict(list)
    initial_reject_counts = Counter()
    for row in pool_rows:
        country = row["country"]
        if row.get("source_row_id") in used_sources:
            continue
        failures, warnings = explicit_audit.audit_explicit_row(row, fetches, ambiguous_ids, fetch_audit)
        if failures:
            initial_reject_counts["audit_failure"] += 1
            continue
        if not high_conf_issue(row, fetches, builder) and not anchor_issue(row, fetches, fetch_audit):
            if warnings:
                generation_source_by_country[country].append(row)
            else:
                strict_existing_by_country[country].append(row)
        elif high_conf_issue(row, fetches, builder) or warnings:
            generation_source_by_country[country].append(row)
        else:
            initial_reject_counts["anchor_issue_not_generation_source"] += 1

    for country in COUNTRY_ORDER:
        strict_existing_by_country[country].sort(key=rank)
        generation_source_by_country[country].sort(key=rank)

    added_existing = []
    for country in COUNTRY_ORDER:
        for row in strict_existing_by_country[country]:
            if counts[country] >= 1000:
                break
            question_key = norm(row.get("question", ""))
            source_id = row.get("source_row_id", "")
            if source_id in used_sources or question_key in used_questions:
                continue
            selected.append(row)
            used_sources.add(source_id)
            used_questions.add(question_key)
            counts[country] += 1
            added_existing.append(row)

    # Use bounded public Wikipedia lead extracts iteratively, only while a
    # country is still short of 1000. This is not a paid LLM/API call; it supplies
    # clean source text for candidate generation without fetching every candidate.
    wiki_extracts: dict[str, str] = {}
    seen_generation_urls: set[str] = set()
    wikipedia_lead_urls_requested = 0

    generated_rows = []
    generation_log = []
    reject_log = []
    for country in COUNTRY_ORDER:
        if counts[country] >= 1000:
            continue
        country_sources = generation_source_by_country[country]
        for batch_start in range(0, len(country_sources), 80):
            if counts[country] >= 1000:
                break
            batch_sources = country_sources[batch_start : batch_start + 80]
            urls_for_batch = []
            for source in batch_sources:
                url = source.get("target_evidence_url", "")
                if url and wiki_title_from_url(url) and url not in seen_generation_urls:
                    seen_generation_urls.add(url)
                    urls_for_batch.append(url)
            if urls_for_batch:
                wikipedia_lead_urls_requested += len(urls_for_batch)
                wiki_extracts.update(fetch_wikipedia_extracts(urls_for_batch))
            for source in batch_sources:
                if counts[country] >= 1000:
                    break
                generated_question, support_sentence = generate_question(source, fetches, wiki_extracts, fetch_audit)
                if not generated_question:
                    reject_log.append(
                        {
                            "country": country,
                            "source_row_id": source.get("source_row_id", ""),
                            "reason": support_sentence,
                            "question": source.get("question", ""),
                            "target_answer": source.get("target_answer", ""),
                            "target_evidence_title": source.get("target_evidence_title", ""),
                        }
                    )
                    continue
                candidate = dict(source)
                original_source = source.get("source_row_id", "")
                candidate["question"] = generated_question
                candidate["localized_question"] = generated_question
                candidate["original_question"] = source.get("question", "")
                candidate["source_row_id"] = f"localnewsqa_explicit_generated_strict_{country.lower().replace(' ', '_')}_{len(generated_rows)+1:04d}"
                candidate["source_split_type"] = "explicit_generated_strict"
                candidate["split_type"] = "explicit"
                candidate["ambiguity_flag"] = False
                candidate["split_family"] = "targetqa"
                candidate["split_name"] = "LocalNewsQA-Explicit-Strict-Defensible-1000-Generated"
                candidate["evidence_hint"] = support_sentence
                candidate["target_evidence_excerpt"] = support_sentence
                if norm(candidate["question"]) in used_questions or candidate["source_row_id"] in used_sources:
                    reject_log.append(
                        {
                            "country": country,
                            "source_row_id": original_source,
                            "reason": "duplicate_generated_question_or_source",
                            "question": generated_question,
                            "target_answer": source.get("target_answer", ""),
                            "target_evidence_title": source.get("target_evidence_title", ""),
                        }
                    )
                    continue
                failures, warnings = explicit_audit.audit_explicit_row(candidate, fetches, ambiguous_ids, fetch_audit)
                if failures or warnings:
                    reject_log.append(
                        {
                            "country": country,
                            "source_row_id": original_source,
                            "reason": "audit_reject",
                            "failures": " | ".join(failures),
                            "warnings": " | ".join(warnings),
                            "question": generated_question,
                            "target_answer": source.get("target_answer", ""),
                            "target_evidence_title": source.get("target_evidence_title", ""),
                        }
                    )
                    continue
                if high_conf_issue(candidate, fetches, builder):
                    reject_log.append(
                        {
                            "country": country,
                            "source_row_id": original_source,
                            "reason": "high_conf_issue_after_generation",
                            "question": generated_question,
                            "target_answer": source.get("target_answer", ""),
                            "target_evidence_title": source.get("target_evidence_title", ""),
                        }
                    )
                    continue
                if anchor_issue(candidate, fetches, fetch_audit):
                    reject_log.append(
                        {
                            "country": country,
                            "source_row_id": original_source,
                            "reason": "anchor_issue_after_generation",
                            "question": generated_question,
                            "target_answer": source.get("target_answer", ""),
                            "target_evidence_title": source.get("target_evidence_title", ""),
                        }
                    )
                    continue
                selected.append(candidate)
                used_sources.add(candidate["source_row_id"])
                used_questions.add(norm(candidate["question"]))
                counts[country] += 1
                generated_rows.append(candidate)
                generation_log.append(
                    {
                        "country": country,
                        "new_source_row_id": candidate["source_row_id"],
                        "source_row_id": original_source,
                        "old_question": source.get("question", ""),
                        "new_question": generated_question,
                        "target_answer": source.get("target_answer", ""),
                        "target_evidence_title": source.get("target_evidence_title", ""),
                        "target_evidence_url": source.get("target_evidence_url", ""),
                        "support_sentence": support_sentence,
                    }
                )

    selected.sort(key=lambda row: (COUNTRY_ORDER.index(row["country"]), rank(row)))
    for idx, row in enumerate(selected, start=1):
        row["id"] = f"localnewsqa_explicit_strict_defensible_1000_{idx:05d}"
        row["split_name"] = "LocalNewsQA-Explicit-Strict-Defensible-1000-Generated"

    source_ids = [row["source_row_id"] for row in selected]
    question_keys = [norm(row["question"]) for row in selected]
    selected_urls = {row.get("target_evidence_url", "") for row in selected if row.get("target_evidence_url", "")}
    validation_errors = []
    for country in COUNTRY_ORDER:
        if counts[country] != 1000:
            validation_errors.append(f"{country}: expected 1000, got {counts[country]}")
    if len(selected) != 17000:
        validation_errors.append(f"expected 17000 rows, got {len(selected)}")
    if len(source_ids) != len(set(source_ids)):
        validation_errors.append("duplicate source ids")
    if len(question_keys) != len(set(question_keys)):
        validation_errors.append("duplicate questions")
    if set(source_ids) & ambiguous_ids:
        validation_errors.append(f"ambiguous overlap: {len(set(source_ids) & ambiguous_ids)}")

    write_jsonl(OUT_JSONL, selected)
    write_csv(OUT_CSV, selected)
    write_csv(GEN_LOG, generation_log)
    write_csv(REJECT_LOG, reject_log)
    write_fetch_cache(FETCH_CACHE, fetches, selected_urls)
    summary = {
        "rows": len(selected),
        "country_counts": dict(Counter(row["country"] for row in selected)),
        "source_split_type_counts": dict(Counter(row.get("source_split_type", "") for row in selected)),
        "added_existing_strict_rows": len(added_existing),
        "generated_rows": len(generated_rows),
        "generated_by_country": dict(Counter(row["country"] for row in generated_rows)),
        "wikipedia_lead_urls_requested": wikipedia_lead_urls_requested,
        "wikipedia_lead_extracts_loaded": len(wiki_extracts),
        "candidate_existing_strict_counts": {country: len(strict_existing_by_country[country]) for country in COUNTRY_ORDER},
        "candidate_generation_source_counts": {
            country: len(generation_source_by_country[country]) for country in COUNTRY_ORDER
        },
        "initial_reject_counts": dict(initial_reject_counts),
        "generation_reject_counts": dict(Counter(row.get("reason", "") for row in reject_log)),
        "duplicate_source_ids": len(source_ids) - len(set(source_ids)),
        "duplicate_questions": len(question_keys) - len(set(question_keys)),
        "ambiguous_overlap": len(set(source_ids) & ambiguous_ids),
        "valid": not validation_errors,
        "validation_errors": validation_errors,
        "paths": {
            "jsonl": str(OUT_JSONL),
            "csv": str(OUT_CSV),
            "fetch_cache": str(FETCH_CACHE),
            "generation_log": str(GEN_LOG),
            "reject_log": str(REJECT_LOG),
            "summary": str(SUMMARY),
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if validation_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
