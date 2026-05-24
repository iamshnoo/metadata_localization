#!/usr/bin/env python3

import csv
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_semantic_gold_ambiguous_1700"
AUDIT_DIR = BASE / "explicit_max_audit"
SOURCE = AUDIT_DIR / "strict_defensible_1000_curated/localnewsqa_targetqa_explicit_strict_defensible_1000_per_country_curated.jsonl"
AMBIGUOUS = BASE / "localnewsqa_ambiguous_semantic_gold_1700.jsonl"
OUTDIR = AUDIT_DIR / "strict_defensible_1000_curated_final"
OUT_JSONL = OUTDIR / "localnewsqa_targetqa_explicit_strict_defensible_1000_per_country_curated_final.jsonl"
OUT_CSV = OUTDIR / "localnewsqa_targetqa_explicit_strict_defensible_1000_per_country_curated_final.csv"
SUMMARY = OUTDIR / "strict_defensible_1000_curated_final_summary.json"
LOG = OUTDIR / "jamaica_manual_curated_additions.csv"
REJECT_LOG = OUTDIR / "jamaica_manual_curated_rejections.csv"
FETCH_CACHE = OUTDIR / "strict_defensible_1000_curated_final_target_evidence_fetches.jsonl"

CURATED_SCRIPT = ROOT / "qa_data/localnewsqa_core/64_build_explicit_strict_defensible_1000_curated.py"
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
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


MANUAL_EXISTING = [
    {
        "source_row_id": "localnewsqa_explicit_0002921",
        "question": "In Jamaica, which person is a Jamaican dancehall reggae artist?",
        "url": "https://en.wikipedia.org/wiki/Busy_Signal",
        "title": "Busy Signal",
        "support": "Reanno Devon Gordon, better known by his stage name Busy Signal, is a Jamaican dancehall reggae artist.",
    },
    {
        "source_row_id": "localnewsqa_explicit_0002923",
        "question": "In Jamaica, which person is a Jamaican reggae singer?",
        "url": "https://en.wikipedia.org/wiki/Gyptian",
        "title": "Gyptian",
        "support": "Windel Beneto Edwards, better known by his stage name Gyptian, is a Jamaican reggae singer.",
    },
    {
        "source_row_id": "localnewsqa_explicit_0002928",
        "question": "In Jamaica, which person is a Jamaican singer and also a yoga teacher?",
        "url": "https://en.wikipedia.org/wiki/Jah9",
        "title": "Jah9",
        "support": "Janine Elizabeth Cunningham, better known as Jah9, is a Jamaican singer and also a yoga teacher.",
    },
    {
        "source_row_id": "localnewsqa_explicit_0002914",
        "question": "In Jamaica, which person is a Jamaican dancehall singer?",
        "url": "https://en.wikipedia.org/wiki/Konshens",
        "title": "Konshens",
        "support": "Garfield Delano Spence, also known as Konshens, is a Jamaican dancehall singer.",
    },
    {
        "source_row_id": "localnewsqa_explicit_0002924",
        "question": "In Jamaica, which person is a reggae and dancehall singer-songwriter from St. Catherine?",
        "url": "https://en.wikipedia.org/wiki/Christopher_Martin_(singer)",
        "title": "Christopher Martin (singer)",
        "support": "Christopher Oteng Martin is a reggae/dancehall singer and songwriter from St. Catherine, Jamaica.",
    },
    {
        "source_row_id": "localnewsqa_explicit_0002905",
        "question": "In Jamaica, which writer is a Jamaican author who won the 2015 Man Booker Prize?",
        "url": "https://en.wikipedia.org/wiki/Marlon_James_(novelist)",
        "title": "Marlon James (novelist)",
        "support": "Marlon James is a Jamaican writer and the author of A Brief History of Seven Killings, which won him the 2015 Man Booker Prize.",
    },
    {
        "source_row_id": "localnewsqa_explicit_0002918",
        "question": "In Jamaica, which writer is the Jamaican author of the novel that won the 2015 Man Booker Prize?",
        "url": "https://en.wikipedia.org/wiki/Marlon_James_(novelist)",
        "title": "Marlon James (novelist)",
        "support": "Marlon James is a Jamaican writer and the author of A Brief History of Seven Killings, which won him the 2015 Man Booker Prize.",
    },
    {
        "source_row_id": "localnewsqa_explicit_0002908",
        "question": "In Jamaica, which person is a Jamaican singer best known for the single Cheerleader?",
        "url": "https://en.wikipedia.org/wiki/Omi_(singer)",
        "title": "Omi (singer)",
        "support": "Omar Samuel Pasley, better known by his stage name Omi, is a Jamaican singer best known for his single Cheerleader.",
    },
    {
        "source_row_id": "localnewsqa_explicit_0002292",
        "question": "In Jamaica, which politician served as Minister of Finance and Planning and Deputy Prime Minister from 2012 to 2016?",
        "url": "https://en.wikipedia.org/wiki/Peter_Phillips_(Jamaican_politician)",
        "title": "Peter Phillips (Jamaican politician)",
        "support": "Peter Phillips is a Jamaican politician who served as Minister of Finance and Planning and Deputy Prime Minister of Jamaica from 2012 to 2016.",
    },
    {
        "source_row_id": "localnewsqa_explicit_0002930",
        "question": "In Jamaica, which person is a Jamaican-American reggae musician who has won two Grammy Awards for Best Reggae Album?",
        "url": "https://en.wikipedia.org/wiki/Shaggy_(musician)",
        "title": "Shaggy (musician)",
        "support": "Orville Richard Burrell, known professionally as Shaggy, is a Jamaican-American reggae musician who has won two Grammy Awards for Best Reggae Album.",
    },
    {
        "source_row_id": "localnewsqa_explicit_0002906",
        "question": "In Jamaica, which person is a Jamaican-American reggae musician awarded the Jamaican Order of Distinction in 2007?",
        "url": "https://en.wikipedia.org/wiki/Shaggy_(musician)",
        "title": "Shaggy (musician)",
        "support": "Orville Richard Burrell, known professionally as Shaggy, is a Jamaican-American reggae musician awarded the Jamaican Order of Distinction in 2007.",
    },
    {
        "source_row_id": "localnewsqa_explicit_0002944",
        "question": "In Jamaica, which person is a Jamaican dancehall recording artist known as the Queen of Dancehall?",
        "url": "https://en.wikipedia.org/wiki/Spice_(musician)",
        "title": "Spice (musician)",
        "support": "Grace Latoya Hamilton, known professionally as Spice, is a Jamaican dancehall recording artist known as the Queen of Dancehall.",
    },
    {
        "source_row_id": "localnewsqa_explicit_0002911",
        "question": "In Jamaica, which person is a Jamaican-American reggae singer and member of the Rastafari movement?",
        "url": "https://en.wikipedia.org/wiki/Tarrus_Riley",
        "title": "Tarrus Riley",
        "support": "Omar Tarrus Riley is a Jamaican-American reggae singer and member of the Rastafari movement.",
    },
    {
        "source_row_id": "localnewsqa_explicit_0002951",
        "question": "In Jamaica, which person is a Jamaican reggae musician and the eldest son of Bob Marley and Rita Marley?",
        "url": "https://en.wikipedia.org/wiki/Ziggy_Marley",
        "title": "Ziggy Marley",
        "support": "David Nesta Ziggy Marley is a Jamaican reggae musician and the eldest son of Bob Marley and Rita Marley.",
    },
    {
        "source_row_id": "localnewsqa_explicit_0002945",
        "question": "In Jamaica, which person is a Jamaican reggae singer whose debut studio album was The Strong One?",
        "url": "https://en.wikipedia.org/wiki/Etana_(musician)",
        "title": "Etana (musician)",
        "support": "Shauna Melissa McKenzie, known by her stage name Etana, is a Jamaican reggae singer whose debut studio album was The Strong One.",
    },
    {
        "source_row_id": "localnewsqa_ambig_0021110",
        "question": "In Jamaica, which person served as Prime Minister of Jamaica from 2012 to 2016?",
        "url": "https://en.wikipedia.org/wiki/Portia_Simpson-Miller",
        "title": "Portia Simpson-Miller",
        "support": "Portia Lucretia Simpson-Miller is a Jamaican politician who served as Prime Minister of Jamaica from 2012 to 2016.",
    },
    {
        "source_row_id": "localnewsqa_ambig_0021041",
        "question": "In Jamaica, which person was the country's first female prime minister?",
        "url": "https://en.wikipedia.org/wiki/Portia_Simpson-Miller",
        "title": "Portia Simpson-Miller",
        "support": "Portia Lucretia Simpson-Miller was Jamaica's first female prime minister.",
    },
    {
        "source_row_id": "localnewsqa_ambig_0021045",
        "question": "In Jamaica, which person led the People's National Party from 2005 to 2017?",
        "url": "https://en.wikipedia.org/wiki/Portia_Simpson-Miller",
        "title": "Portia Simpson-Miller",
        "support": "Portia Lucretia Simpson-Miller was leader of the People's National Party from 2005 to 2017.",
    },
    {
        "source_row_id": "localnewsqa_ambig_0021046",
        "question": "In Jamaica, which person served as Prime Minister of Jamaica in two separate periods?",
        "url": "https://en.wikipedia.org/wiki/Portia_Simpson-Miller",
        "title": "Portia Simpson-Miller",
        "support": "Portia Lucretia Simpson-Miller served as Prime Minister of Jamaica from 2006 to 2007 and from 2012 to 2016.",
    },
    {
        "source_row_id": "localnewsqa_ambig_0021506",
        "question": "In Jamaica, the island name Xaymaca is glossed as Land of Wood and what?",
        "url": "https://en.wikipedia.org/wiki/Jamaica",
        "title": "Jamaica",
        "support": "The Indigenous peoples called the island Xaymaca, meaning the Land of Wood and Water.",
    },
    {
        "source_row_id": "localnewsqa_explicit_0002806",
        "question": "In Jamaica, which ministry's official website is titled Ministry of Education, Skills, Youth and Information?",
        "url": "https://moey.gov.jm/",
        "title": "Ministry of Education Jamaica",
        "support": "Ministry of Education, Skills, Youth and Information, Jamaica.",
    },
    {
        "source_row_id": "localnewsqa_explicit_0002879",
        "question": "In Jamaica, which government ministry publishes education outcomes and school exam resources on its official site?",
        "url": "https://moey.gov.jm/",
        "title": "Ministry of Education Jamaica",
        "support": "Ministry of Education, Skills, Youth and Information, Jamaica, publishes education outcomes and external exam resources.",
    },
    {
        "source_row_id": "localnewsqa_explicit_0002760",
        "question": "In Jamaica, which ministry has an official site carrying the slogan Every Child Can Learn, Every Child Must Learn?",
        "url": "https://moey.gov.jm/",
        "title": "Ministry of Education Jamaica",
        "support": "Ministry of Education, Skills, Youth and Information, Jamaica: Every Child Can Learn, Every Child Must Learn.",
    },
    {
        "source_row_id": "localnewsqa_explicit_0002606",
        "question": "In Jamaica, which prosecutorial office's website says Crown prosecutors provide a public service?",
        "url": "https://dpp.gov.jm/",
        "title": "Office of the Director of Public Prosecutions Jamaica",
        "support": "Office of the Director of Public Prosecutions Jamaica: A Crown prosecutor is a provider of a public service.",
    },
    {
        "source_row_id": "localnewsqa_explicit_0002419",
        "question": "In Jamaica, which office describes itself as a voice of the voiceless and handles complaints against public authorities?",
        "url": "https://opd.gov.jm/",
        "title": "Public Defender Jamaica",
        "support": "Government of Jamaica Office of the Public Defender describes itself as a voice of the voiceless.",
    },
    {
        "source_row_id": "localnewsqa_explicit_0002717",
        "question": "In Jamaica, which company describes itself as Future Energy Source Company and provides premium fuel services?",
        "url": "https://www.fescoja.com/",
        "title": "FESCO Jamaica",
        "support": "FESCO - Future Energy Source Company provides premium fuel services in Jamaica.",
    },
    {
        "source_row_id": "localnewsqa_explicit_0002731",
        "question": "In Jamaica, which company is described as offering airport food and beverage concessions at Sangster International Airport?",
        "url": "https://www.jamaicaobserver.com/2025/10/10/express-caterings-us5-m-gamble/",
        "title": "EXPRESS CATERING'S US$5-M GAMBLE - Jamaica Observer",
        "support": "The Guitar Bar at the Sangster International Airport is part of Express Catering Limited's offerings at the airport.",
    },
    {
        "source_row_id": "localnewsqa_explicit_0002740",
        "question": "In Jamaica, which examination had results released by the Jamaica Information Service in the cited education item?",
        "url": "https://jis.gov.jm/radio_programs/gsat-results-released-today/",
        "title": "GSAT Results Released Jamaica",
        "support": "GSAT Results To Be Released Today, Jamaica Information Service.",
    },
    {
        "source_row_id": "localnewsqa_ambig_0021255",
        "question": "In Jamaica, which market is listed by the Jamaica Stock Exchange alongside the Main Market, USD Market, and Bond Market?",
        "url": "https://en.wikipedia.org/wiki/Jamaica_Stock_Exchange",
        "title": "Jamaica Stock Exchange",
        "support": "The Jamaica Stock Exchange is composed of multiple markets: Main Market, Junior Market, USD Market, and Bond Market.",
    },
]

MANUAL_NEW = [
    {
        "source_row_id": "localnewsqa_explicit_manual_jamaica_kingston_0001",
        "question": "In Jamaica, which city is the capital and largest city?",
        "target_answer": "Kingston",
        "options": ["Kingston", "Montego Bay", "Spanish Town", "Mandeville"],
        "url": "https://en.wikipedia.org/wiki/Kingston,_Jamaica",
        "title": "Kingston, Jamaica",
        "support": "Kingston is the capital and largest city of Jamaica.",
    },
    {
        "source_row_id": "localnewsqa_explicit_manual_jamaica_holness_0002",
        "question": "In Jamaica, which politician became prime minister in 2016 after the general election?",
        "target_answer": "Andrew Holness",
        "options": ["Andrew Holness", "Portia Simpson-Miller", "Bruce Golding", "P. J. Patterson"],
        "url": "https://en.wikipedia.org/wiki/Andrew_Holness",
        "title": "Andrew Holness",
        "support": "Andrew Holness is a Jamaican politician who has served as Prime Minister of Jamaica since 2016.",
    },
    {
        "source_row_id": "localnewsqa_explicit_manual_jamaica_currency_0003",
        "question": "In Jamaica, what is the national currency?",
        "target_answer": "Jamaican dollar",
        "options": ["Jamaican dollar", "United States dollar", "East Caribbean dollar", "Barbadian dollar"],
        "url": "https://en.wikipedia.org/wiki/Jamaican_dollar",
        "title": "Jamaican dollar",
        "support": "The Jamaican dollar is the currency of Jamaica.",
    },
    {
        "source_row_id": "localnewsqa_explicit_manual_jamaica_jcf_0004",
        "question": "In Jamaica, which organization is the national police force?",
        "target_answer": "Jamaica Constabulary Force",
        "options": ["Jamaica Constabulary Force", "Jamaica Defence Force", "Island Special Constabulary Force", "Major Organised Crime and Anti-Corruption Agency"],
        "url": "https://en.wikipedia.org/wiki/Jamaica_Constabulary_Force",
        "title": "Jamaica Constabulary Force",
        "support": "The Jamaica Constabulary Force is the national official police force of Jamaica.",
    },
    {
        "source_row_id": "localnewsqa_explicit_manual_jamaica_jdf_0005",
        "question": "In Jamaica, which organization is the country's combined military?",
        "target_answer": "Jamaica Defence Force",
        "options": ["Jamaica Defence Force", "Jamaica Constabulary Force", "Jamaica Fire Brigade", "Jamaica Customs Agency"],
        "url": "https://en.wikipedia.org/wiki/Jamaica_Defence_Force",
        "title": "Jamaica Defence Force",
        "support": "The Jamaica Defence Force is the combined military of Jamaica.",
    },
]


def option_parts(options: Any) -> list[str]:
    if isinstance(options, list):
        return [str(option) for option in options]
    return [part.strip() for part in str(options or "").split("||") if part.strip()]


def ensure_fetch(url: str, fetches: dict[str, dict], fetch_audit: Any, cur_mod: Any) -> None:
    if fetches.get(url, {}).get("ok") and int(fetches[url].get("text_len", 0) or 0) >= 500:
        return
    fetched = fetch_audit.fetch_url(url, timeout=25, delay=0)
    if fetched.get("ok"):
        cur_mod.merge_curation_fetch(fetches, url, fetched)
    if cur_mod.wiki_title_from_url(url):
        wiki = cur_mod.fetch_wikipedia_leads([url]).get(url, {})
        if wiki.get("ok"):
            cur_mod.merge_curation_fetch(fetches, url, wiki)


def main() -> None:
    cur_mod = load_module(CURATED_SCRIPT, "cur64")
    explicit_audit = load_module(AUDIT_EXPLICIT_SCRIPT, "audit48")
    fetch_audit = load_module(FETCH_AUDIT_SCRIPT, "fetch32")
    builder = load_module(BUILDER_SCRIPT, "builder45")

    rows = read_jsonl(SOURCE)
    ambiguous_ids = {row["source_row_id"] for row in read_jsonl(AMBIGUOUS)}
    fetches: dict[str, dict] = {}
    for cache in cur_mod.FETCH_CACHES:
        if cache == cur_mod.CANDIDATE_FETCH_CACHE:
            continue
        fetches.update(explicit_audit.load_fetch_cache(cache))
    for url, fetch in explicit_audit.load_fetch_cache(cur_mod.CANDIDATE_FETCH_CACHE).items():
        cur_mod.merge_curation_fetch(fetches, url, fetch)

    pool_by_source = {}
    for pool in cur_mod.CANDIDATE_POOLS:
        for row in read_jsonl(pool):
            pool_by_source.setdefault(row.get("source_row_id", ""), row)

    template = next(row for row in rows if row.get("country") == "Jamaica")
    used_sources = {row.get("source_row_id", "") for row in rows}
    used_questions = {norm(row.get("question", "")) for row in rows}
    counts = Counter(row["country"] for row in rows)

    added = []
    rejected = []

    def try_add(candidate: dict, entry: dict) -> None:
        if counts["Jamaica"] >= 1000:
            return
        if candidate.get("source_row_id", "") in used_sources:
            return
        if norm(candidate.get("question", "")) in used_questions:
            rejected.append({**entry, "reason": "duplicate_question"})
            return
        ensure_fetch(candidate["target_evidence_url"], fetches, fetch_audit, cur_mod)
        failures, warnings = explicit_audit.audit_explicit_row(candidate, fetches, ambiguous_ids, fetch_audit)
        if cur_mod.high_conf_issue(candidate, fetches, builder):
            failures = failures + ["high_confidence_question_cue"]
        if cur_mod.anchor_issue(candidate, fetches, fetch_audit):
            failures = failures + ["answer_anchor_issue"]
        if failures or warnings:
            rejected.append(
                {
                    **entry,
                    "reason": "audit_reject",
                    "failures": " | ".join(failures),
                    "warnings": " | ".join(warnings),
                }
            )
            return
        rows.append(candidate)
        used_sources.add(candidate["source_row_id"])
        used_questions.add(norm(candidate["question"]))
        counts["Jamaica"] += 1
        added.append(
            {
                "source_row_id": candidate["source_row_id"],
                "question": candidate["question"],
                "target_answer": candidate["target_answer"],
                "target_evidence_title": candidate["target_evidence_title"],
                "target_evidence_url": candidate["target_evidence_url"],
                "support": candidate["target_evidence_excerpt"],
            }
        )

    for entry in MANUAL_EXISTING:
        source = pool_by_source.get(entry["source_row_id"])
        if not source:
            rejected.append({**entry, "reason": "source_not_found"})
            continue
        candidate = dict(source)
        candidate["question"] = entry["question"]
        candidate["localized_question"] = entry["question"]
        candidate["original_question"] = source.get("question", "")
        candidate["target_evidence_url"] = entry["url"]
        candidate["target_evidence_title"] = entry["title"]
        candidate["target_evidence_excerpt"] = entry["support"]
        candidate["evidence_hint"] = entry["support"]
        candidate["split_type"] = "explicit"
        candidate["ambiguity_flag"] = False
        candidate["split_family"] = "targetqa"
        candidate["split_name"] = "LocalNewsQA-Explicit-Strict-Defensible-1000-Curated-Final"
        candidate["curation_status"] = "manual_jamaica_strict_final"
        try_add(candidate, entry)

    for entry in MANUAL_NEW:
        candidate = dict(template)
        candidate.update(
            {
                "source_row_id": entry["source_row_id"],
                "country": "Jamaica",
                "target_country": "Jamaica",
                "contrast_country": "Canada",
                "question": entry["question"],
                "localized_question": entry["question"],
                "original_question": "",
                "options": entry["options"],
                "correct_answer": entry["target_answer"],
                "target_answer": entry["target_answer"],
                "distractors": [option for option in entry["options"] if norm(option) != norm(entry["target_answer"])],
                "contrast_answer": "",
                "target_evidence_url": entry["url"],
                "target_evidence_title": entry["title"],
                "target_evidence_excerpt": entry["support"],
                "evidence_hint": entry["support"],
                "source_split_type": "explicit_manual_curated",
                "split_type": "explicit",
                "ambiguity_flag": False,
                "split_family": "targetqa",
                "split_name": "LocalNewsQA-Explicit-Strict-Defensible-1000-Curated-Final",
                "curation_status": "manual_jamaica_new_strict_final",
            }
        )
        try_add(candidate, entry)

    rows.sort(key=lambda row: (COUNTRY_ORDER.index(row["country"]), norm(row.get("target_answer", "")), norm(row.get("question", "")), row.get("source_row_id", "")))
    for idx, row in enumerate(rows, start=1):
        row["id"] = f"localnewsqa_explicit_strict_defensible_1000_{idx:05d}"
        row["split_name"] = "LocalNewsQA-Explicit-Strict-Defensible-1000-Curated-Final"

    source_ids = [row["source_row_id"] for row in rows]
    question_keys = [norm(row["question"]) for row in rows]
    selected_urls = {row.get("target_evidence_url", "") for row in rows if row.get("target_evidence_url", "")}
    validation_errors = []
    country_counts = Counter(row["country"] for row in rows)
    for country in COUNTRY_ORDER:
        if country_counts[country] != 1000:
            validation_errors.append(f"{country}: expected 1000, got {country_counts[country]}")
    if len(rows) != 17000:
        validation_errors.append(f"expected 17000 rows, got {len(rows)}")
    if len(source_ids) != len(set(source_ids)):
        validation_errors.append("duplicate source ids")
    if len(question_keys) != len(set(question_keys)):
        validation_errors.append("duplicate questions")
    if set(source_ids) & ambiguous_ids:
        validation_errors.append(f"ambiguous overlap: {len(set(source_ids) & ambiguous_ids)}")

    write_jsonl(OUT_JSONL, rows)
    write_csv(OUT_CSV, rows)
    write_csv(LOG, added)
    write_csv(REJECT_LOG, rejected)
    write_jsonl(FETCH_CACHE, [fetches[url] for url in sorted(selected_urls) if url in fetches])
    summary = {
        "rows": len(rows),
        "country_counts": dict(country_counts),
        "added_rows": len(added),
        "added_by_source_split_type": dict(Counter(row.get("source_split_type", "") for row in rows if row.get("curation_status") in {"manual_jamaica_strict_final", "manual_jamaica_new_strict_final"})),
        "rejected_rows": len(rejected),
        "rejected_reasons": dict(Counter(row.get("reason", "") for row in rejected)),
        "duplicate_source_ids": len(source_ids) - len(set(source_ids)),
        "duplicate_questions": len(question_keys) - len(set(question_keys)),
        "ambiguous_overlap": len(set(source_ids) & ambiguous_ids),
        "valid": not validation_errors,
        "validation_errors": validation_errors,
        "paths": {
            "jsonl": str(OUT_JSONL),
            "csv": str(OUT_CSV),
            "fetch_cache": str(FETCH_CACHE),
            "additions": str(LOG),
            "rejections": str(REJECT_LOG),
            "summary": str(SUMMARY),
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if validation_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
