#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_semantic_gold_ambiguous_1700"
AUDIT_DIR = BASE / "explicit_max_audit"
FINAL_DIR = AUDIT_DIR / "strict_defensible_1000_curated_final"
DEFAULT_INPUT = FINAL_DIR / "localnewsqa_targetqa_explicit_strict_defensible_1000_per_country_curated_final.jsonl"
DEFAULT_CSV = FINAL_DIR / "localnewsqa_targetqa_explicit_strict_defensible_1000_per_country_curated_final.csv"
DEFAULT_FETCH_CACHE = FINAL_DIR / "strict_defensible_1000_curated_final_target_evidence_fetches.jsonl"
DEFAULT_WARNINGS = FINAL_DIR / "audit/explicit_max_quality_warnings.csv"
AMBIGUOUS = BASE / "localnewsqa_ambiguous_semantic_gold_1700.jsonl"
SUMMARY = FINAL_DIR / "warning_replacement_summary.json"
REPLACEMENT_LOG = FINAL_DIR / "warning_replacement_log.csv"
REJECT_LOG = FINAL_DIR / "warning_replacement_reject_log.csv"

AUDIT_EXPLICIT_SCRIPT = ROOT / "qa_data/localnewsqa_core/48_audit_explicit_max_split.py"
FETCH_AUDIT_SCRIPT = ROOT / "qa_data/localnewsqa_core/32_web_audit_ambiguous_verifiable.py"
BUILDER_SCRIPT = ROOT / "qa_data/localnewsqa_core/45_build_relation_strict_gold_ambiguous.py"
CURATED_SCRIPT = ROOT / "qa_data/localnewsqa_core/64_build_explicit_strict_defensible_1000_curated.py"

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
    "United Kingdom": ["united kingdom", "british", "england", "scotland", "wales"],
    "United States": ["united states", "american"],
}

CONTINENT = {
    "Bangladesh": "Asia",
    "Canada": "North America",
    "Ghana": "Africa",
    "Hong Kong": "Asia",
    "India": "Asia",
    "Ireland": "Europe",
    "Jamaica": "North America",
    "Kenya": "Africa",
    "Malaysia": "Asia",
    "Nigeria": "Africa",
    "Pakistan": "Asia",
    "Philippines": "Asia",
    "South Africa": "Africa",
    "Sri Lanka": "Asia",
    "Tanzania": "Africa",
    "United Kingdom": "Europe",
    "United States": "North America",
}

MANUAL_SEEDS = [
    {
        "country": "Bangladesh",
        "answer": "Bangladesh Bank",
        "title": "Bangladesh Bank",
        "question": "In Bangladesh, which institution is the central bank?",
        "options": ["Bangladesh Bank", "Dhaka Stock Exchange", "University of Dhaka", "Bangladesh Television"],
        "topic": "Business and economy",
    },
    {
        "country": "Bangladesh",
        "answer": "Dhaka",
        "title": "Dhaka",
        "question": "In Bangladesh, which city is the capital and largest city?",
        "options": ["Dhaka", "Chittagong", "Khulna", "Sylhet"],
        "topic": "Places",
    },
    {
        "country": "Canada",
        "answer": "Bank of Canada",
        "title": "Bank of Canada",
        "question": "In Canada, which institution is the country's central bank?",
        "options": ["Bank of Canada", "Toronto Stock Exchange", "CBC", "Statistics Canada"],
        "topic": "Business and economy",
    },
    {
        "country": "Canada",
        "answer": "Ottawa",
        "title": "Ottawa",
        "question": "In Canada, which city is the federal capital?",
        "options": ["Ottawa", "Toronto", "Vancouver", "Montreal"],
        "topic": "Places",
    },
    {
        "country": "Ghana",
        "answer": "Bank of Ghana",
        "title": "Bank of Ghana",
        "question": "In Ghana, which institution is the central bank?",
        "options": ["Bank of Ghana", "Ghana Stock Exchange", "University of Ghana", "Accra Sports Stadium"],
        "topic": "Business and economy",
    },
    {
        "country": "Ghana",
        "answer": "Ghana Stock Exchange",
        "title": "Ghana Stock Exchange",
        "question": "In Ghana, which institution is the principal stock exchange?",
        "options": ["Ghana Stock Exchange", "Bank of Ghana", "Daily Graphic", "University of Ghana"],
        "topic": "Business and economy",
    },
    {
        "country": "Ghana",
        "answer": "University of Ghana",
        "title": "University of Ghana",
        "question": "In Ghana, which public university is located in Legon, Accra?",
        "options": ["University of Ghana", "Bank of Ghana", "Ghana Stock Exchange", "Accra Sports Stadium"],
        "topic": "Education",
    },
    {
        "country": "Ghana",
        "answer": "Accra",
        "title": "Accra",
        "question": "In Ghana, which city is the capital and largest city?",
        "options": ["Accra", "Kumasi", "Tamale", "Cape Coast"],
        "topic": "Places",
    },
    {
        "country": "India",
        "answer": "Reserve Bank of India",
        "title": "Reserve Bank of India",
        "question": "In India, which institution is the central bank and banking regulator?",
        "options": ["Reserve Bank of India", "Bombay Stock Exchange", "Parliament of India", "University of Delhi"],
        "topic": "Business and economy",
    },
    {
        "country": "India",
        "answer": "Parliament of India",
        "title": "Parliament of India",
        "question": "In India, which institution is the supreme legislative body?",
        "options": ["Parliament of India", "Reserve Bank of India", "Election Commission of India", "University of Delhi"],
        "topic": "Government",
    },
    {
        "country": "Ireland",
        "answer": "Central Bank of Ireland",
        "title": "Central Bank of Ireland",
        "question": "In Ireland, which institution is the country's central bank?",
        "options": ["Central Bank of Ireland", "Bank of Ireland", "Dublin City University", "Raidio Teilifis Eireann"],
        "topic": "Business and economy",
    },
    {
        "country": "Ireland",
        "answer": "Dublin",
        "title": "Dublin",
        "question": "In Ireland, which city is the capital and largest city?",
        "options": ["Dublin", "Cork", "Galway", "Limerick"],
        "topic": "Places",
    },
    {
        "country": "Jamaica",
        "answer": "Norman Manley International Airport",
        "title": "Norman Manley International Airport",
        "question": "In Jamaica, which international airport serves Kingston?",
        "options": ["Norman Manley International Airport", "Sangster International Airport", "Bob Marley Museum", "Jamaica Stock Exchange"],
        "topic": "Transport",
    },
    {
        "country": "Jamaica",
        "answer": "Sangster International Airport",
        "title": "Sangster International Airport",
        "question": "In Jamaica, which international airport is located east of Montego Bay?",
        "options": ["Sangster International Airport", "Norman Manley International Airport", "The Gleaner", "Emancipation Park"],
        "topic": "Transport",
    },
    {
        "country": "Jamaica",
        "answer": "Emancipation Park",
        "title": "Emancipation Park (Kingston, Jamaica)",
        "question": "In Jamaica, which public park is located in Kingston?",
        "options": ["Emancipation Park", "Bob Marley Museum", "Jamaica Observer", "Jamaica Defence Force"],
        "topic": "Places",
    },
    {
        "country": "Jamaica",
        "answer": "Jamaica national football team",
        "title": "Jamaica national football team",
        "question": "In Jamaica, which national team represents the country in men's international football?",
        "options": ["Jamaica national football team", "Jamaica national cricket team", "Jamaica Labour Party", "Jamaica Stock Exchange"],
        "topic": "Sports",
    },
    {
        "country": "Jamaica",
        "answer": "Jamaica national cricket team",
        "title": "Jamaica national cricket team",
        "question": "In Jamaica, which national team is the country's representative cricket team?",
        "options": ["Jamaica national cricket team", "Jamaica national football team", "Jamaica Defence Force", "Jamaica Observer"],
        "topic": "Sports",
    },
    {
        "country": "Jamaica",
        "answer": "Bob Marley Museum",
        "title": "Bob Marley Museum",
        "question": "In Jamaica, which Kingston tourist attraction is dedicated to Bob Marley?",
        "options": ["Bob Marley Museum", "Emancipation Park", "Jamaica Observer", "Jamaica Stock Exchange"],
        "topic": "Arts and culture",
    },
    {
        "country": "Jamaica",
        "answer": "Mandeville",
        "title": "Mandeville, Jamaica",
        "question": "In Jamaica, which town is the capital of Manchester parish?",
        "options": ["Mandeville", "Kingston", "Ocho Rios", "Montego Bay"],
        "topic": "Places",
    },
    {
        "country": "Jamaica",
        "answer": "Ocho Rios",
        "title": "Ocho Rios",
        "question": "In Jamaica, which town is in Saint Ann parish on the north coast?",
        "options": ["Ocho Rios", "Mandeville", "Kingston", "Portmore"],
        "topic": "Places",
    },
    {
        "country": "Kenya",
        "answer": "Central Bank of Kenya",
        "title": "Central Bank of Kenya",
        "question": "In Kenya, which institution is the monetary authority?",
        "options": ["Central Bank of Kenya", "Nairobi Securities Exchange", "Nation Media Group", "University of Nairobi"],
        "topic": "Business and economy",
    },
    {
        "country": "Kenya",
        "answer": "Nairobi",
        "title": "Nairobi",
        "question": "In Kenya, which city is the capital and largest city?",
        "options": ["Nairobi", "Mombasa", "Kisumu", "Nakuru"],
        "topic": "Places",
    },
    {
        "country": "Malaysia",
        "answer": "Bank Negara Malaysia",
        "title": "Bank Negara Malaysia",
        "question": "In Malaysia, which institution is the central bank?",
        "options": ["Bank Negara Malaysia", "Bursa Malaysia", "Petronas", "Parliament of Malaysia"],
        "topic": "Business and economy",
    },
    {
        "country": "Malaysia",
        "answer": "Bursa Malaysia",
        "title": "Bursa Malaysia",
        "question": "In Malaysia, which institution is the country's stock exchange?",
        "options": ["Bursa Malaysia", "Bank Negara Malaysia", "Petronas", "University of Malaya"],
        "topic": "Business and economy",
    },
    {
        "country": "Malaysia",
        "answer": "Parliament of Malaysia",
        "title": "Parliament of Malaysia",
        "question": "In Malaysia, which institution is the national legislature?",
        "options": ["Parliament of Malaysia", "Bursa Malaysia", "Bank Negara Malaysia", "Petronas"],
        "topic": "Government",
    },
    {
        "country": "Malaysia",
        "answer": "Petronas",
        "title": "Petronas",
        "question": "In Malaysia, which oil and gas company is wholly owned by the government?",
        "options": ["Petronas", "Bursa Malaysia", "Bank Negara Malaysia", "Parliament of Malaysia"],
        "topic": "Business and economy",
    },
    {
        "country": "Nigeria",
        "answer": "Central Bank of Nigeria",
        "title": "Central Bank of Nigeria",
        "question": "In Nigeria, which institution is the central bank and apex monetary authority?",
        "options": ["Central Bank of Nigeria", "Nigerian Stock Exchange", "University of Lagos", "National Assembly"],
        "topic": "Business and economy",
    },
    {
        "country": "Nigeria",
        "answer": "Abuja",
        "title": "Abuja",
        "question": "In Nigeria, which city is the capital city?",
        "options": ["Abuja", "Lagos", "Kano", "Ibadan"],
        "topic": "Places",
    },
    {
        "country": "Pakistan",
        "answer": "State Bank of Pakistan",
        "title": "State Bank of Pakistan",
        "question": "In Pakistan, which institution is the central bank?",
        "options": ["State Bank of Pakistan", "Pakistan Stock Exchange", "National Assembly of Pakistan", "University of Karachi"],
        "topic": "Business and economy",
    },
    {
        "country": "Pakistan",
        "answer": "Pakistan Stock Exchange",
        "title": "Pakistan Stock Exchange",
        "question": "In Pakistan, which stock exchange has trading floors in Karachi, Islamabad, and Lahore?",
        "options": ["Pakistan Stock Exchange", "State Bank of Pakistan", "National Assembly of Pakistan", "Dawn"],
        "topic": "Business and economy",
    },
    {
        "country": "Pakistan",
        "answer": "Islamabad",
        "title": "Islamabad",
        "question": "In Pakistan, which city is the capital city?",
        "options": ["Islamabad", "Lahore", "Karachi", "Peshawar"],
        "topic": "Places",
    },
    {
        "country": "Philippines",
        "answer": "Bangko Sentral ng Pilipinas",
        "title": "Bangko Sentral ng Pilipinas",
        "question": "In the Philippines, which institution is the central bank?",
        "options": ["Bangko Sentral ng Pilipinas", "Department of Education", "BusinessMirror", "University of the Philippines"],
        "topic": "Business and economy",
    },
    {
        "country": "Philippines",
        "answer": "Manila",
        "title": "Manila",
        "question": "In the Philippines, which city is the capital city?",
        "options": ["Manila", "Quezon City", "Davao City", "Cebu City"],
        "topic": "Places",
    },
    {
        "country": "Philippines",
        "answer": "Quezon City",
        "title": "Quezon City",
        "question": "In the Philippines, which city is the most populous city?",
        "options": ["Quezon City", "Manila", "Davao City", "Tagaytay"],
        "topic": "Places",
    },
    {
        "country": "South Africa",
        "answer": "South African Reserve Bank",
        "title": "South African Reserve Bank",
        "question": "In South Africa, which institution is the central bank?",
        "options": ["South African Reserve Bank", "Johannesburg Stock Exchange", "Parliament of South Africa", "University of Cape Town"],
        "topic": "Business and economy",
    },
    {
        "country": "South Africa",
        "answer": "Pretoria",
        "title": "Pretoria",
        "question": "In South Africa, which city is the administrative capital?",
        "options": ["Pretoria", "Cape Town", "Johannesburg", "Durban"],
        "topic": "Places",
    },
    {
        "country": "South Africa",
        "answer": "Johannesburg",
        "title": "Johannesburg",
        "question": "In South Africa, which city is the most populous city?",
        "options": ["Johannesburg", "Pretoria", "Cape Town", "Durban"],
        "topic": "Places",
    },
]


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


def norm(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_warning_ids(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["source_row_id"] for row in csv.DictReader(handle)}


def fetch_wikipedia_pages(titles: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    api_url = "https://en.wikipedia.org/w/api.php"
    for start in range(0, len(titles), 10):
        batch = titles[start : start + 10]
        response = requests.get(
            api_url,
            params={
                "action": "query",
                "format": "json",
                "prop": "extracts|info",
                "explaintext": 1,
                "exintro": 0,
                "exlimit": "max",
                "redirects": 1,
                "inprop": "url",
                "titles": "|".join(batch),
            },
            timeout=25,
            headers={"User-Agent": "LocalNewsQA-final-warning-replacements/1.0"},
        )
        response.raise_for_status()
        data = response.json().get("query", {})
        redirects = {item.get("from", ""): item.get("to", "") for item in data.get("redirects", [])}
        normalized = {item.get("from", ""): item.get("to", "") for item in data.get("normalized", [])}
        pages = {page.get("title", ""): page for page in data.get("pages", {}).values()}
        for original_title in batch:
            title = redirects.get(original_title, normalized.get(original_title, original_title))
            title = redirects.get(title, normalized.get(title, title))
            page = pages.get(title) or pages.get(original_title)
            text = re.sub(r"\s+", " ", str(page.get("extract", "") if page else "")).strip()
            final_title = page.get("title", title) if page else title
            url_title = quote(str(final_title).replace(" ", "_"), safe="()_,'.")
            out[original_title] = {
                "url": f"https://en.wikipedia.org/wiki/{url_title}",
                "ok": bool(text),
                "status_code": "200" if text else "404",
                "final_url": page.get("fullurl", f"https://en.wikipedia.org/wiki/{url_title}") if page else "",
                "content_type": "application/json",
                "title": final_title,
                "text": text[:120_000],
                "text_len": len(text[:120_000]),
                "error": "" if text else "wikipedia_api_empty_extract",
                "elapsed_sec": 0.0,
            }
        time.sleep(0.05)
    return out


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\[[^\]]*\]", "", str(text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return [part.strip(" .") for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def support_sentence(seed: dict, fetch: dict, fetch_audit: Any) -> str:
    aliases = list(fetch_audit.answer_aliases(seed["answer"])) + [seed["answer"], seed["title"], fetch.get("title", "")]
    markers = COUNTRY_MARKERS[seed["country"]]
    for sentence in split_sentences(fetch.get("text", ""))[:12]:
        sent_norm = fetch_audit.normalize_text(sentence)
        has_answer = fetch_audit.contains_any(sent_norm, aliases) or norm(fetch.get("title", "")) == norm(seed["answer"])
        has_country = fetch_audit.contains_any(sent_norm, markers)
        if has_answer and has_country and 30 <= len(sentence) <= 500:
            if fetch_audit.contains_any(sent_norm, aliases):
                return sentence
            return f"{fetch.get('title', seed['answer'])}. {sentence}"
    return ""


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
            str(fetch.get("api_text_for_curation", ""))[:1500],
        ]
    )
    return not fetch_audit.contains_any(fetch_audit.normalize_text(anchor_text), fetch_audit.answer_aliases(row.get("target_answer", "")))


def row_is_strict(row: dict, fetches: dict[str, dict], ambiguous_ids: set[str], explicit_audit: Any, fetch_audit: Any, builder: Any) -> tuple[bool, list[str], list[str]]:
    failures, warnings = explicit_audit.audit_explicit_row(row, fetches, ambiguous_ids, fetch_audit)
    if high_conf_issue(row, fetches, builder):
        failures = failures + ["high_confidence_question_cue"]
    if anchor_issue(row, fetches, fetch_audit):
        failures = failures + ["answer_anchor_issue"]
    return not failures and not warnings, failures, warnings


def make_manual_row(seed: dict, source_id: str, support: str, fetch: dict) -> dict:
    return {
        "id": "",
        "source_row_id": source_id,
        "country": seed["country"],
        "continent": CONTINENT[seed["country"]],
        "target_country": seed["country"],
        "contrast_country": "United States" if seed["country"] != "United States" else "Canada",
        "topic": seed["topic"],
        "year": "2026",
        "question": seed["question"],
        "localized_question": seed["question"],
        "original_question": seed["question"],
        "options": seed["options"],
        "correct_answer": seed["answer"],
        "distractors": [option for option in seed["options"] if norm(option) != norm(seed["answer"])],
        "target_answer": seed["answer"],
        "contrast_answer": "",
        "evidence_hint": support,
        "target_evidence_url": fetch["url"],
        "target_evidence_title": fetch["title"],
        "target_evidence_excerpt": support,
        "contrast_evidence_url": "",
        "contrast_evidence_title": "",
        "contrast_evidence_excerpt": "",
        "semantic_risk_band": "manual_gold",
        "semantic_error_probability": "0.0000",
        "evidence_support_score": "1.0000",
        "issue_families": "",
        "split_name": "LocalNewsQA-Explicit-Strict-Defensible-1000-Curated-Final",
        "split_type": "explicit",
        "split_family": "targetqa",
        "ambiguity_flag": False,
        "source_split_type": "explicit_manual_curated_replacement",
        "curation_status": "manual_warning_replacement_wikipedia_supported",
        "curation_support_sentence": support,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace final explicit rows that still have reviewer-visible warnings.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--fetch-cache", type=Path, default=DEFAULT_FETCH_CACHE)
    parser.add_argument("--warnings", type=Path, default=DEFAULT_WARNINGS)
    args = parser.parse_args()

    explicit_audit = load_module(AUDIT_EXPLICIT_SCRIPT, "audit48")
    fetch_audit = load_module(FETCH_AUDIT_SCRIPT, "fetch32")
    builder = load_module(BUILDER_SCRIPT, "builder45")
    cur_mod = load_module(CURATED_SCRIPT, "cur64")

    rows = read_jsonl(args.input)
    warning_ids = load_warning_ids(args.warnings)
    removed_rows = [row for row in rows if row.get("source_row_id", "") in warning_ids]
    kept_rows = [row for row in rows if row.get("source_row_id", "") not in warning_ids]
    ambiguous_ids = {row["source_row_id"] for row in read_jsonl(AMBIGUOUS)}

    fetches: dict[str, dict] = {}
    for cache in [
        args.fetch_cache,
        FINAL_DIR / "audit/explicit_target_evidence_fetches.jsonl",
        *cur_mod.FETCH_CACHES,
    ]:
        fetches.update(explicit_audit.load_fetch_cache(cache))

    used_sources = {row.get("source_row_id", "") for row in kept_rows}
    used_questions = {norm(row.get("question", "")) for row in kept_rows}
    deficits = {country: 1000 - Counter(row["country"] for row in kept_rows)[country] for country in COUNTRY_ORDER}
    additions = []
    rejects = []

    candidate_pools = [
        FINAL_DIR / "audit/localnewsqa_targetqa_explicit_style_max_strict_no_warnings.jsonl",
        AUDIT_DIR / "strict_defensible_1000_curated/audit/localnewsqa_targetqa_explicit_style_max_strict_no_warnings.jsonl",
        *cur_mod.CANDIDATE_POOLS,
    ]
    seen_pool_sources = set()
    for pool in candidate_pools:
        if not pool.exists():
            continue
        for candidate in read_jsonl(pool):
            country = candidate.get("country", "")
            if deficits.get(country, 0) <= 0:
                continue
            source_id = candidate.get("source_row_id", "")
            question_key = norm(candidate.get("question", ""))
            if (
                not source_id
                or source_id in seen_pool_sources
                or source_id in used_sources
                or source_id in ambiguous_ids
                or question_key in used_questions
            ):
                continue
            seen_pool_sources.add(source_id)
            candidate = dict(candidate)
            candidate["split_name"] = "LocalNewsQA-Explicit-Strict-Defensible-1000-Curated-Final"
            ok, failures, warnings = row_is_strict(candidate, fetches, ambiguous_ids, explicit_audit, fetch_audit, builder)
            if not ok:
                rejects.append(
                    {
                        "source": "candidate_pool",
                        "pool": str(pool),
                        "country": country,
                        "source_row_id": source_id,
                        "target_answer": candidate.get("target_answer", ""),
                        "reason": "audit_reject",
                        "failures": " | ".join(failures),
                        "warnings": " | ".join(warnings),
                    }
                )
                continue
            kept_rows.append(candidate)
            additions.append({**candidate, "replacement_source": "candidate_pool", "replaced_country_deficit": country})
            used_sources.add(source_id)
            used_questions.add(question_key)
            deficits[country] -= 1

    remaining_countries = {country for country, deficit in deficits.items() if deficit > 0}
    needed_titles = sorted({seed["title"] for seed in MANUAL_SEEDS if seed["country"] in remaining_countries})
    wiki_fetches = fetch_wikipedia_pages(needed_titles) if needed_titles else {}
    manual_index_by_country = defaultdict(int)
    for seed in MANUAL_SEEDS:
        country = seed["country"]
        if deficits.get(country, 0) <= 0:
            continue
        fetch = wiki_fetches.get(seed["title"], {})
        if not fetch.get("ok"):
            rejects.append({"source": "manual_seed", "country": country, "answer": seed["answer"], "reason": "fetch_failed"})
            continue
        support = support_sentence(seed, fetch, fetch_audit)
        if not support:
            rejects.append({"source": "manual_seed", "country": country, "answer": seed["answer"], "reason": "no_support_sentence"})
            continue
        fetches[fetch["url"]] = fetch
        source_id = f"localnewsqa_explicit_manual_warning_replacement_{norm(country).replace(' ', '_')}_{manual_index_by_country[country] + 1:04d}"
        manual_index_by_country[country] += 1
        question_key = norm(seed["question"])
        if source_id in used_sources or question_key in used_questions:
            rejects.append({"source": "manual_seed", "country": country, "answer": seed["answer"], "reason": "duplicate_source_or_question"})
            continue
        candidate = make_manual_row(seed, source_id, support, fetch)
        ok, failures, warnings = row_is_strict(candidate, fetches, ambiguous_ids, explicit_audit, fetch_audit, builder)
        if not ok:
            rejects.append(
                {
                    "source": "manual_seed",
                    "country": country,
                    "answer": seed["answer"],
                    "reason": "audit_reject",
                    "failures": " | ".join(failures),
                    "warnings": " | ".join(warnings),
                    "support": support,
                }
            )
            continue
        kept_rows.append(candidate)
        additions.append({**candidate, "replacement_source": "manual_seed", "replaced_country_deficit": country})
        used_sources.add(source_id)
        used_questions.add(question_key)
        deficits[country] -= 1

    kept_rows.sort(
        key=lambda row: (
            COUNTRY_ORDER.index(row["country"]),
            norm(row.get("target_answer", "")),
            norm(row.get("question", "")),
            row.get("source_row_id", ""),
        )
    )
    for idx, row in enumerate(kept_rows, start=1):
        row["id"] = f"localnewsqa_explicit_strict_defensible_1000_{idx:05d}"
        row["split_name"] = "LocalNewsQA-Explicit-Strict-Defensible-1000-Curated-Final"

    country_counts = Counter(row["country"] for row in kept_rows)
    source_ids = [row.get("source_row_id", "") for row in kept_rows]
    question_keys = [norm(row.get("question", "")) for row in kept_rows]
    selected_urls = {row.get("target_evidence_url", "") for row in kept_rows if row.get("target_evidence_url", "")}
    validation_errors = []
    if len(kept_rows) != 17_000:
        validation_errors.append(f"expected 17000 rows, got {len(kept_rows)}")
    for country in COUNTRY_ORDER:
        if country_counts[country] != 1000:
            validation_errors.append(f"{country}: expected 1000, got {country_counts[country]}")
    if len(source_ids) != len(set(source_ids)):
        validation_errors.append("duplicate source ids")
    if len(question_keys) != len(set(question_keys)):
        validation_errors.append("duplicate questions")
    if set(source_ids) & ambiguous_ids:
        validation_errors.append(f"ambiguous overlap: {len(set(source_ids) & ambiguous_ids)}")
    if any(deficits.values()):
        validation_errors.append(f"unfilled deficits: {dict(deficits)}")

    write_jsonl(args.input, kept_rows)
    write_csv(args.csv, kept_rows)
    write_jsonl(args.fetch_cache, [fetches[url] for url in sorted(selected_urls) if url in fetches])
    write_csv(REPLACEMENT_LOG, additions)
    write_csv(REJECT_LOG, rejects)

    summary = {
        "rows": len(kept_rows),
        "removed_warning_rows": len(removed_rows),
        "removed_by_country": dict(Counter(row.get("country", "") for row in removed_rows)),
        "replacement_rows_added": len(additions),
        "replacement_by_source": dict(Counter(row.get("replacement_source", "") for row in additions)),
        "replacement_by_country": dict(Counter(row.get("country", "") for row in additions)),
        "reject_rows": len(rejects),
        "reject_reasons": dict(Counter(row.get("reason", "") for row in rejects)),
        "country_counts": dict(country_counts),
        "source_split_type_counts": dict(Counter(row.get("source_split_type", "") for row in kept_rows)),
        "curation_status_counts": dict(Counter(row.get("curation_status", "") for row in kept_rows)),
        "duplicate_source_ids": len(source_ids) - len(set(source_ids)),
        "duplicate_questions": len(question_keys) - len(set(question_keys)),
        "ambiguous_overlap": len(set(source_ids) & ambiguous_ids),
        "valid": not validation_errors,
        "validation_errors": validation_errors,
        "paths": {
            "jsonl": str(args.input),
            "csv": str(args.csv),
            "fetch_cache": str(args.fetch_cache),
            "replacement_log": str(REPLACEMENT_LOG),
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
