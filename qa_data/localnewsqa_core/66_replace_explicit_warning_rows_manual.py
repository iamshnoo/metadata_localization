#!/usr/bin/env python3

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
OUTDIR = AUDIT_DIR / "strict_defensible_1000_curated"
DATASET = OUTDIR / "localnewsqa_targetqa_explicit_strict_defensible_1000_per_country_curated.jsonl"
CSV_OUT = OUTDIR / "localnewsqa_targetqa_explicit_strict_defensible_1000_per_country_curated.csv"
WARNINGS = OUTDIR / "audit/explicit_max_quality_warnings.csv"
FETCH_CACHE = OUTDIR / "strict_defensible_1000_curated_target_evidence_fetches.jsonl"
SUMMARY = OUTDIR / "strict_defensible_1000_curated_build_summary.json"
REPLACEMENT_LOG = OUTDIR / "manual_warning_replacement_log.csv"
REJECT_LOG = OUTDIR / "manual_warning_replacement_reject_log.csv"
AMBIGUOUS = BASE / "localnewsqa_ambiguous_semantic_gold_1700.jsonl"

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

COUNTRY_MARKERS = {
    "Bangladesh": ["bangladesh", "bangladeshi"],
    "Canada": ["canada", "canadian"],
    "Ghana": ["ghana", "ghanaian"],
    "India": ["india", "indian"],
    "Ireland": ["ireland", "irish"],
    "Jamaica": ["jamaica", "jamaican", "kingston"],
    "Kenya": ["kenya", "kenyan"],
    "Malaysia": ["malaysia", "malaysian"],
    "Nigeria": ["nigeria", "nigerian"],
    "Pakistan": ["pakistan", "pakistani"],
    "Philippines": ["philippines", "philippine", "filipino"],
    "South Africa": ["south africa", "south african"],
}

CONTINENT = {
    "Bangladesh": "Asia",
    "Canada": "North America",
    "Ghana": "Africa",
    "India": "Asia",
    "Ireland": "Europe",
    "Jamaica": "North America",
    "Kenya": "Africa",
    "Malaysia": "Asia",
    "Nigeria": "Africa",
    "Pakistan": "Asia",
    "Philippines": "Asia",
    "South Africa": "Africa",
}

SEEDS = [
    {
        "country": "Bangladesh",
        "answer": "Dhaka",
        "title": "Dhaka",
        "question": "In Bangladesh, which city is formerly known as Dacca and is the capital and largest city of Bangladesh?",
        "options": ["Dhaka", "Chittagong", "Khulna", "Sylhet"],
        "topic": "Places",
    },
    {
        "country": "Bangladesh",
        "answer": "Bangladesh Bank",
        "title": "Bangladesh Bank",
        "question": "In Bangladesh, which institution is the central bank of Bangladesh?",
        "options": ["Bangladesh Bank", "Bangladesh Television", "University of Dhaka", "Dhaka Stock Exchange"],
        "topic": "Business and economy",
    },
    {
        "country": "Canada",
        "answer": "Ottawa",
        "title": "Ottawa",
        "question": "In Canada, which city is the capital city of Canada?",
        "options": ["Ottawa", "Toronto", "Vancouver", "Montreal"],
        "topic": "Places",
    },
    {
        "country": "Canada",
        "answer": "Bank of Canada",
        "title": "Bank of Canada",
        "question": "In Canada, which institution is the central bank of Canada?",
        "options": ["Bank of Canada", "Toronto Stock Exchange", "CBC", "Statistics Canada"],
        "topic": "Business and economy",
    },
    {
        "country": "Ghana",
        "answer": "Bank of Ghana",
        "title": "Bank of Ghana",
        "question": "In Ghana, which institution is the central bank of Ghana?",
        "options": ["Bank of Ghana", "Ghana Stock Exchange", "University of Ghana", "Accra Sports Stadium"],
        "topic": "Business and economy",
    },
    {
        "country": "Ghana",
        "answer": "Ghana Stock Exchange",
        "title": "Ghana Stock Exchange",
        "question": "In Ghana, which institution is the principal stock exchange of Ghana?",
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
        "question": "In Ghana, which city is the capital and largest city of Ghana?",
        "options": ["Accra", "Kumasi", "Tamale", "Cape Coast"],
        "topic": "Places",
    },
    {
        "country": "India",
        "answer": "Ministry of Agriculture and Farmers' Welfare",
        "title": "Ministry of Agriculture and Farmers' Welfare",
        "question": "In India, which ministry is responsible for agriculture and farmers' welfare?",
        "options": [
            "Ministry of Agriculture and Farmers' Welfare",
            "Reserve Bank of India",
            "Election Commission of India",
            "University of Delhi",
        ],
        "topic": "Government",
    },
    {
        "country": "India",
        "answer": "Reserve Bank of India",
        "title": "Reserve Bank of India",
        "question": "In India, which institution is the central bank and regulatory body for India's banking system?",
        "options": [
            "Reserve Bank of India",
            "Ministry of Agriculture and Farmers' Welfare",
            "Bombay Stock Exchange",
            "Parliament of India",
        ],
        "topic": "Business and economy",
    },
    {
        "country": "Ireland",
        "answer": "Dublin",
        "title": "Dublin",
        "question": "In Ireland, which city is the capital and largest city of Ireland?",
        "options": ["Dublin", "Cork", "Galway", "Limerick"],
        "topic": "Places",
    },
    {
        "country": "Ireland",
        "answer": "Central Bank of Ireland",
        "title": "Central Bank of Ireland",
        "question": "In Ireland, which institution is the central bank of Ireland?",
        "options": ["Central Bank of Ireland", "Dublin City University", "Raidio Teilifis Eireann", "Bank of Ireland"],
        "topic": "Business and economy",
    },
    {
        "country": "Jamaica",
        "answer": "Norman Manley International Airport",
        "title": "Norman Manley International Airport",
        "question": "In Jamaica, which international airport serves Kingston, Jamaica?",
        "options": ["Norman Manley International Airport", "Sangster International Airport", "Bob Marley Museum", "Jamaica Stock Exchange"],
        "topic": "Transport",
    },
    {
        "country": "Jamaica",
        "answer": "Sangster International Airport",
        "title": "Sangster International Airport",
        "question": "In Jamaica, which international airport is located east of Montego Bay, Jamaica?",
        "options": ["Sangster International Airport", "Norman Manley International Airport", "The Gleaner", "Emancipation Park"],
        "topic": "Transport",
    },
    {
        "country": "Jamaica",
        "answer": "Emancipation Park",
        "title": "Emancipation Park (Kingston, Jamaica)",
        "question": "In Jamaica, which public park is located in Kingston, Jamaica?",
        "options": ["Emancipation Park", "Bob Marley Museum", "Jamaica Observer", "Jamaica Defence Force"],
        "topic": "Places",
    },
    {
        "country": "Jamaica",
        "answer": "Jamaica national football team",
        "title": "Jamaica national football team",
        "question": "In Jamaica, which national team represents Jamaica in men's international football?",
        "options": ["Jamaica national football team", "Jamaica national cricket team", "Jamaica Labour Party", "Jamaica Stock Exchange"],
        "topic": "Sports",
    },
    {
        "country": "Jamaica",
        "answer": "Jamaica national cricket team",
        "title": "Jamaica national cricket team",
        "question": "In Jamaica, which national team is the representative cricket team of Jamaica?",
        "options": ["Jamaica national cricket team", "Jamaica national football team", "Jamaica Defence Force", "Jamaica Observer"],
        "topic": "Sports",
    },
    {
        "country": "Jamaica",
        "answer": "Bob Marley Museum",
        "title": "Bob Marley Museum",
        "question": "In Jamaica, which tourist attraction in Kingston is dedicated to reggae musician Bob Marley?",
        "options": ["Bob Marley Museum", "Emancipation Park", "Jamaica Observer", "Jamaica Stock Exchange"],
        "topic": "Arts and culture",
    },
    {
        "country": "Jamaica",
        "answer": "Mandeville",
        "title": "Mandeville, Jamaica",
        "question": "In Jamaica, which town is the capital and largest town in the parish of Manchester?",
        "options": ["Mandeville", "Kingston", "Ocho Rios", "Montego Bay"],
        "topic": "Places",
    },
    {
        "country": "Jamaica",
        "answer": "Ocho Rios",
        "title": "Ocho Rios",
        "question": "In Jamaica, which town is in the parish of Saint Ann on the north coast of Jamaica?",
        "options": ["Ocho Rios", "Mandeville", "Kingston", "Portmore"],
        "topic": "Places",
    },
    {
        "country": "Jamaica",
        "answer": "Montego Bay",
        "title": "Montego Bay",
        "question": "In Jamaica, which city is also known locally as MoBay?",
        "options": ["Montego Bay", "Kingston", "Mandeville", "Ocho Rios"],
        "topic": "Places",
    },
    {
        "country": "Jamaica",
        "answer": "People's National Party",
        "title": "People's National Party (Jamaica)",
        "question": "In Jamaica, which political party was founded in 1938 by Norman Washington Manley?",
        "options": ["People's National Party", "Jamaica Labour Party", "Jamaica Stock Exchange", "Jamaica Defence Force"],
        "topic": "Politics",
    },
    {
        "country": "Jamaica",
        "answer": "Jamaica Labour Party",
        "title": "Jamaica Labour Party",
        "question": "In Jamaica, which political party is described as one of the two major political parties in Jamaica?",
        "options": ["Jamaica Labour Party", "People's National Party", "Jamaica Observer", "Jamaica Stock Exchange"],
        "topic": "Politics",
    },
    {
        "country": "Jamaica",
        "answer": "Jamaica national football team",
        "title": "Jamaica national football team",
        "question": "In Jamaica, which national team is governed by the Jamaica Football Federation?",
        "options": ["Jamaica national football team", "Jamaica national cricket team", "Jamaica Labour Party", "Bob Marley Museum"],
        "topic": "Sports",
    },
    {
        "country": "Kenya",
        "answer": "Nairobi",
        "title": "Nairobi",
        "question": "In Kenya, which city is the capital and largest city of Kenya?",
        "options": ["Nairobi", "Mombasa", "Kisumu", "Nakuru"],
        "topic": "Places",
    },
    {
        "country": "Kenya",
        "answer": "Central Bank of Kenya",
        "title": "Central Bank of Kenya",
        "question": "In Kenya, which institution is the monetary authority of Kenya?",
        "options": ["Central Bank of Kenya", "Nairobi Securities Exchange", "Nation Media Group", "University of Nairobi"],
        "topic": "Business and economy",
    },
    {
        "country": "Malaysia",
        "answer": "Bank Negara Malaysia",
        "title": "Bank Negara Malaysia",
        "question": "In Malaysia, which institution is the central bank of Malaysia?",
        "options": ["Bank Negara Malaysia", "Bursa Malaysia", "Petronas", "Parliament of Malaysia"],
        "topic": "Business and economy",
    },
    {
        "country": "Malaysia",
        "answer": "Bursa Malaysia",
        "title": "Bursa Malaysia",
        "question": "In Malaysia, which institution is the stock exchange of Malaysia?",
        "options": ["Bursa Malaysia", "Bank Negara Malaysia", "Petronas", "University of Malaya"],
        "topic": "Business and economy",
    },
    {
        "country": "Malaysia",
        "answer": "Parliament of Malaysia",
        "title": "Parliament of Malaysia",
        "question": "In Malaysia, which institution is the national legislature of Malaysia?",
        "options": ["Parliament of Malaysia", "Bursa Malaysia", "Bank Negara Malaysia", "Petronas"],
        "topic": "Government",
    },
    {
        "country": "Malaysia",
        "answer": "Petronas",
        "title": "Petronas",
        "question": "In Malaysia, which oil and gas company is wholly owned by the government of Malaysia?",
        "options": ["Petronas", "Bursa Malaysia", "Bank Negara Malaysia", "Parliament of Malaysia"],
        "topic": "Business and economy",
    },
    {
        "country": "Malaysia",
        "answer": "Kuala Lumpur",
        "title": "Kuala Lumpur",
        "question": "In Malaysia, which city is the capital and largest city of Malaysia?",
        "options": ["Kuala Lumpur", "George Town", "Johor Bahru", "Kuching"],
        "topic": "Places",
    },
    {
        "country": "Nigeria",
        "answer": "Abuja",
        "title": "Abuja",
        "question": "In Nigeria, which city is the capital city of Nigeria?",
        "options": ["Abuja", "Lagos", "Kano", "Ibadan"],
        "topic": "Places",
    },
    {
        "country": "Nigeria",
        "answer": "Central Bank of Nigeria",
        "title": "Central Bank of Nigeria",
        "question": "In Nigeria, which institution is the central bank and apex monetary authority of Nigeria?",
        "options": ["Central Bank of Nigeria", "Nigerian Stock Exchange", "University of Lagos", "National Assembly"],
        "topic": "Business and economy",
    },
    {
        "country": "Pakistan",
        "answer": "Islamabad",
        "title": "Islamabad",
        "question": "In Pakistan, which city is the capital city of Pakistan?",
        "options": ["Islamabad", "Lahore", "Karachi", "Peshawar"],
        "topic": "Places",
    },
    {
        "country": "Pakistan",
        "answer": "State Bank of Pakistan",
        "title": "State Bank of Pakistan",
        "question": "In Pakistan, which institution is the central bank of Pakistan?",
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
        "country": "Philippines",
        "answer": "Manila",
        "title": "Manila",
        "question": "In the Philippines, which city is the capital city of the Philippines?",
        "options": ["Manila", "Quezon City", "Davao City", "Cebu City"],
        "topic": "Places",
    },
    {
        "country": "Philippines",
        "answer": "Bangko Sentral ng Pilipinas",
        "title": "Bangko Sentral ng Pilipinas",
        "question": "In the Philippines, which institution is the central bank of the Philippines?",
        "options": ["Bangko Sentral ng Pilipinas", "Department of Education", "BusinessMirror", "University of the Philippines"],
        "topic": "Business and economy",
    },
    {
        "country": "Philippines",
        "answer": "Quezon City",
        "title": "Quezon City",
        "question": "In the Philippines, which city is the most populous city in the Philippines?",
        "options": ["Quezon City", "Manila", "Davao City", "Tagaytay"],
        "topic": "Places",
    },
    {
        "country": "South Africa",
        "answer": "Pretoria",
        "title": "Pretoria",
        "question": "In South Africa, which city is the administrative capital of South Africa?",
        "options": ["Pretoria", "Cape Town", "Johannesburg", "Durban"],
        "topic": "Places",
    },
    {
        "country": "South Africa",
        "answer": "South African Reserve Bank",
        "title": "South African Reserve Bank",
        "question": "In South Africa, which institution is the central bank of South Africa?",
        "options": ["South African Reserve Bank", "Johannesburg Stock Exchange", "Parliament of South Africa", "University of Cape Town"],
        "topic": "Business and economy",
    },
    {
        "country": "South Africa",
        "answer": "Johannesburg",
        "title": "Johannesburg",
        "question": "In South Africa, which city is the most populous city in South Africa?",
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
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def csv_value(value: Any) -> str:
    if isinstance(value, list):
        return " || ".join(str(part) for part in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return "" if value is None else str(value)


def write_dataset_csv(path: Path, rows: list[dict]) -> None:
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


def write_csv(path: Path, rows: list[dict]) -> None:
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
        writer.writerows(rows)


def norm(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_wikipedia_pages(titles: list[str]) -> dict[str, dict]:
    out = {}
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
            timeout=20,
            headers={"User-Agent": "LocalNewsQA-warning-row-replacements/1.0"},
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
    for sentence in split_sentences(fetch.get("text", ""))[:10]:
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
        ]
    )
    return not fetch_audit.contains_any(fetch_audit.normalize_text(anchor_text), fetch_audit.answer_aliases(row["target_answer"]))


def make_row(seed: dict, source_id: str, support: str, fetch: dict) -> dict:
    return {
        "id": "",
        "source_row_id": source_id,
        "country": seed["country"],
        "continent": CONTINENT[seed["country"]],
        "target_country": seed["country"],
        "contrast_country": "United States" if seed["country"] != "United States" else "Canada",
        "topic": seed["topic"],
        "year": "2015",
        "question": seed["question"],
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
        "split_name": "LocalNewsQA-Explicit-Strict-Defensible-1000-Curated",
        "split_type": "explicit",
        "split_family": "LocalNewsQA-Core",
        "ambiguity_flag": False,
        "source_split_type": "explicit_manual_curated_replacement",
        "localized_question": seed["question"],
        "curation_status": "manual_warning_replacement_wikipedia_supported",
        "curation_support_sentence": support,
    }


def main() -> None:
    explicit_audit = load_module(AUDIT_EXPLICIT_SCRIPT, "audit48")
    fetch_audit = load_module(FETCH_AUDIT_SCRIPT, "fetch32")
    builder = load_module(BUILDER_SCRIPT, "builder45")

    rows = read_jsonl(DATASET)
    warning_rows = list(csv.DictReader(WARNINGS.open()))
    remove_ids = {row["source_row_id"] for row in warning_rows}
    rows = [row for row in rows if row["source_row_id"] not in remove_ids]
    ambiguous_ids = {row["source_row_id"] for row in read_jsonl(AMBIGUOUS)}
    fetches = explicit_audit.load_fetch_cache(FETCH_CACHE)
    page_fetches = fetch_wikipedia_pages(sorted({seed["title"] for seed in SEEDS}))

    used_sources = {row["source_row_id"] for row in rows}
    used_questions = {norm(row["question"]) for row in rows}
    deficits = {country: 1000 - Counter(row["country"] for row in rows)[country] for country in COUNTRY_ORDER}
    additions = []
    rejects = []
    seed_index_by_country = defaultdict(int)
    for seed in SEEDS:
        country = seed["country"]
        if deficits.get(country, 0) <= 0:
            continue
        fetch = page_fetches.get(seed["title"], {})
        if not fetch.get("ok"):
            rejects.append({"country": country, "answer": seed["answer"], "reason": "fetch_failed"})
            continue
        source_id = f"localnewsqa_explicit_manual_replacement_{norm(country).replace(' ', '_')}_{seed_index_by_country[country] + 1:04d}"
        seed_index_by_country[country] += 1
        if source_id in used_sources or norm(seed["question"]) in used_questions:
            rejects.append({"country": country, "answer": seed["answer"], "reason": "duplicate_source_or_question"})
            continue
        support = support_sentence(seed, fetch, fetch_audit)
        if not support:
            rejects.append({"country": country, "answer": seed["answer"], "reason": "no_support_sentence"})
            continue
        fetches[fetch["url"]] = fetch
        row = make_row(seed, source_id, support, fetch)
        failures, warnings = explicit_audit.audit_explicit_row(row, fetches, ambiguous_ids, fetch_audit)
        hc = high_conf_issue(row, fetches, builder)
        ai = anchor_issue(row, fetches, fetch_audit)
        if failures or warnings or hc or ai:
            rejects.append(
                {
                    "country": country,
                    "answer": seed["answer"],
                    "reason": "audit_reject",
                    "failures": " | ".join(failures),
                    "warnings": " | ".join(warnings),
                    "high_conf_issue": hc,
                    "anchor_issue": ai,
                    "support": support,
                }
            )
            continue
        rows.append(row)
        additions.append(row)
        used_sources.add(source_id)
        used_questions.add(norm(row["question"]))
        deficits[country] -= 1

    validation_errors = []
    country_counts = Counter(row["country"] for row in rows)
    for country in COUNTRY_ORDER:
        if country_counts[country] != 1000:
            validation_errors.append(f"{country}: expected 1000, got {country_counts[country]}")
    if len(rows) != 17000:
        validation_errors.append(f"expected 17000 rows, got {len(rows)}")
    source_ids = [row["source_row_id"] for row in rows]
    question_keys = [norm(row["question"]) for row in rows]
    if len(source_ids) != len(set(source_ids)):
        validation_errors.append("duplicate source ids")
    if len(question_keys) != len(set(question_keys)):
        validation_errors.append("duplicate questions")
    if set(source_ids) & ambiguous_ids:
        validation_errors.append(f"ambiguous overlap: {len(set(source_ids) & ambiguous_ids)}")

    rows.sort(key=lambda row: (COUNTRY_ORDER.index(row["country"]), norm(row.get("target_answer", "")), norm(row["question"])))
    for idx, row in enumerate(rows, start=1):
        row["id"] = f"localnewsqa_explicit_strict_defensible_1000_{idx:05d}"
        row["split_name"] = "LocalNewsQA-Explicit-Strict-Defensible-1000-Curated"

    write_jsonl(DATASET, rows)
    write_dataset_csv(CSV_OUT, rows)
    write_csv(
        REPLACEMENT_LOG,
        [
            {
                "source_row_id": row["source_row_id"],
                "country": row["country"],
                "question": row["question"],
                "target_answer": row["target_answer"],
                "target_evidence_title": row["target_evidence_title"],
                "target_evidence_url": row["target_evidence_url"],
                "support_sentence": row["target_evidence_excerpt"],
            }
            for row in additions
        ],
    )
    write_csv(REJECT_LOG, rejects)
    write_jsonl(FETCH_CACHE, [fetches[url] for url in sorted(fetches) if fetches[url].get("ok")])

    summary = {
        "rows": len(rows),
        "country_counts": dict(country_counts),
        "removed_warning_rows": len(remove_ids),
        "manual_replacement_rows_added": len(additions),
        "manual_replacement_rejects": len(rejects),
        "source_split_type_counts": dict(Counter(row.get("source_split_type", "") for row in rows)),
        "duplicate_source_ids": len(source_ids) - len(set(source_ids)),
        "duplicate_questions": len(question_keys) - len(set(question_keys)),
        "ambiguous_overlap": len(set(source_ids) & ambiguous_ids),
        "valid": not validation_errors,
        "validation_errors": validation_errors,
        "paths": {
            "jsonl": str(DATASET),
            "csv": str(CSV_OUT),
            "fetch_cache": str(FETCH_CACHE),
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
