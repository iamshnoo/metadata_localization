#!/usr/bin/env python3

import csv
import importlib.util
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_semantic_gold_ambiguous_1700"
AUDIT_DIR = BASE / "explicit_max_audit"
OUTDIR = AUDIT_DIR / "strict_defensible_1000_curated"
INPUT = OUTDIR / "localnewsqa_targetqa_explicit_strict_defensible_1000_per_country_curated.jsonl"
OUT_JSONL = INPUT
OUT_CSV = OUTDIR / "localnewsqa_targetqa_explicit_strict_defensible_1000_per_country_curated.csv"
SUMMARY = OUTDIR / "strict_defensible_1000_curated_build_summary.json"
FILL_LOG = OUTDIR / "jamaica_manual_fill_log.csv"
REJECT_LOG = OUTDIR / "jamaica_manual_fill_reject_log.csv"
FETCH_CACHE = OUTDIR / "strict_defensible_1000_curated_target_evidence_fetches.jsonl"
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

SEEDS = [
    {
        "answer": "Bob Marley",
        "title": "Bob Marley",
        "question": "In Jamaica, which person was a Jamaican singer, musician, and songwriter who helped pioneer reggae?",
        "options": ["Bob Marley", "Usain Bolt", "Portia Simpson-Miller", "Marcus Garvey"],
        "topic": "Arts and culture",
        "year": "1970",
    },
    {
        "answer": "Usain Bolt",
        "title": "Usain Bolt",
        "question": "In Jamaica, which person is a Jamaican retired sprinter widely considered one of the greatest sprinters of all time?",
        "options": ["Usain Bolt", "Jimmy Cliff", "Marlon James", "Andrew Holness"],
        "topic": "Sports",
        "year": "2017",
    },
    {
        "answer": "Shelly-Ann Fraser-Pryce",
        "title": "Shelly-Ann Fraser-Pryce",
        "question": "In Jamaica, which person is a Jamaican track and field sprinter who competes in the 60 metres, 100 metres, and 200 metres?",
        "options": ["Shelly-Ann Fraser-Pryce", "Elaine Thompson-Herah", "Louise Bennett-Coverley", "Grace Jones"],
        "topic": "Sports",
        "year": "2022",
    },
    {
        "answer": "Elaine Thompson-Herah",
        "title": "Elaine Thompson-Herah",
        "question": "In Jamaica, which person is a Jamaican sprinter and five-time Olympic champion?",
        "options": ["Elaine Thompson-Herah", "Merlene Ottey", "Koffee", "Lorna Goodison"],
        "topic": "Sports",
        "year": "2021",
    },
    {
        "answer": "Veronica Campbell Brown",
        "title": "Veronica Campbell Brown",
        "question": "In Jamaica, which person is a former Jamaican track and field sprinter who specialized in the 100 metres and 200 metres?",
        "options": ["Veronica Campbell Brown", "Asafa Powell", "Sean Paul", "P. J. Patterson"],
        "topic": "Sports",
        "year": "2012",
    },
    {
        "answer": "Asafa Powell",
        "title": "Asafa Powell",
        "question": "In Jamaica, which person is a Jamaican former sprinter who specialized in the 100 metres?",
        "options": ["Asafa Powell", "Yohan Blake", "Damian Marley", "Michael Manley"],
        "topic": "Sports",
        "year": "2015",
    },
    {
        "answer": "Yohan Blake",
        "title": "Yohan Blake",
        "question": "In Jamaica, which person is a Jamaican sprinter who won the 100 metres at the 2011 World Championships?",
        "options": ["Yohan Blake", "Ziggy Marley", "Claude McKay", "Andrew Holness"],
        "topic": "Sports",
        "year": "2011",
    },
    {
        "answer": "Merlene Ottey",
        "title": "Merlene Ottey",
        "question": "In Jamaica, which person is a Jamaican-Slovenian former track and field sprinter?",
        "options": ["Merlene Ottey", "Veronica Campbell Brown", "Grace Jones", "Lorna Goodison"],
        "topic": "Sports",
        "year": "2000",
    },
    {
        "answer": "Portia Simpson-Miller",
        "title": "Portia Simpson-Miller",
        "question": "In Jamaica, which person served as prime minister from 2006 to 2007 and again from 2012 to 2016?",
        "options": ["Portia Simpson-Miller", "P. J. Patterson", "Michael Manley", "Andrew Holness"],
        "topic": "Politics",
        "year": "2016",
    },
    {
        "answer": "Andrew Holness",
        "title": "Andrew Holness",
        "question": "In Jamaica, which politician is described as the ninth prime minister of Jamaica?",
        "options": ["Andrew Holness", "Michael Manley", "Marcus Garvey", "Jimmy Cliff"],
        "topic": "Politics",
        "year": "2016",
    },
    {
        "answer": "P. J. Patterson",
        "title": "P. J. Patterson",
        "question": "In Jamaica, which person served as prime minister from 1992 to 2006?",
        "options": ["P. J. Patterson", "Andrew Holness", "Bob Marley", "Marlon James"],
        "topic": "Politics",
        "year": "2006",
    },
    {
        "answer": "Michael Manley",
        "title": "Michael Manley",
        "question": "In Jamaica, which politician served as the fourth prime minister of Jamaica?",
        "options": ["Michael Manley", "P. J. Patterson", "Usain Bolt", "Claude McKay"],
        "topic": "Politics",
        "year": "1980",
    },
    {
        "answer": "Marcus Garvey",
        "title": "Marcus Garvey",
        "question": "In Jamaica, which political activist founded the Universal Negro Improvement Association?",
        "options": ["Marcus Garvey", "Michael Manley", "Bob Marley", "Jimmy Cliff"],
        "topic": "Public life and history",
        "year": "1920",
    },
    {
        "answer": "Louise Bennett-Coverley",
        "title": "Louise Bennett-Coverley",
        "question": "In Jamaica, which poet and folklorist wrote and performed in Jamaican Patois?",
        "options": ["Louise Bennett-Coverley", "Lorna Goodison", "Grace Jones", "Koffee"],
        "topic": "Arts and culture",
        "year": "1960",
    },
    {
        "answer": "Marlon James",
        "title": "Marlon James (novelist)",
        "question": "In Jamaica, which writer won the 2015 Man Booker Prize for A Brief History of Seven Killings?",
        "options": ["Marlon James", "Claude McKay", "Marcus Garvey", "Sean Paul"],
        "topic": "Arts and culture",
        "year": "2015",
    },
    {
        "answer": "Claude McKay",
        "title": "Claude McKay",
        "question": "In Jamaica, which writer and poet was a central figure in the Harlem Renaissance?",
        "options": ["Claude McKay", "Marlon James", "Portia Simpson-Miller", "Asafa Powell"],
        "topic": "Arts and culture",
        "year": "1920",
    },
    {
        "answer": "Lorna Goodison",
        "title": "Lorna Goodison",
        "question": "In Jamaica, which poet was appointed Poet Laureate of Jamaica in 2017?",
        "options": ["Lorna Goodison", "Louise Bennett-Coverley", "Spice", "Grace Jones"],
        "topic": "Arts and culture",
        "year": "2017",
    },
    {
        "answer": "Jimmy Cliff",
        "title": "Jimmy Cliff",
        "question": "In Jamaica, which person was a Jamaican ska, rocksteady, reggae, and soul musician?",
        "options": ["Jimmy Cliff", "Buju Banton", "P. J. Patterson", "Usain Bolt"],
        "topic": "Arts and culture",
        "year": "1972",
    },
    {
        "answer": "Buju Banton",
        "title": "Buju Banton",
        "question": "In Jamaica, which person is a Jamaican reggae and dancehall musician?",
        "options": ["Buju Banton", "Busy Signal", "Andrew Holness", "Marlon James"],
        "topic": "Arts and culture",
        "year": "2018",
    },
    {
        "answer": "Shaggy",
        "title": "Shaggy (musician)",
        "question": "In Jamaica, which person is a Jamaican-American reggae rapper, singer, and songwriter?",
        "options": ["Shaggy", "Sean Paul", "Yohan Blake", "Claude McKay"],
        "topic": "Arts and culture",
        "year": "2001",
    },
    {
        "answer": "Sean Paul",
        "title": "Sean Paul",
        "question": "In Jamaica, which person is a Jamaican dancehall deejay, singer, and rapper?",
        "options": ["Sean Paul", "Shaggy", "Asafa Powell", "Portia Simpson-Miller"],
        "topic": "Arts and culture",
        "year": "2003",
    },
    {
        "answer": "Grace Jones",
        "title": "Grace Jones",
        "question": "In Jamaica, which person is a Jamaican-American model, singer, songwriter, record producer, and actress?",
        "options": ["Grace Jones", "Koffee", "Shelly-Ann Fraser-Pryce", "Louise Bennett-Coverley"],
        "topic": "Arts and culture",
        "year": "1981",
    },
    {
        "answer": "Koffee",
        "title": "Koffee (singer)",
        "question": "In Jamaica, which person is a Jamaican singer, songwriter, rapper, deejay, and guitarist?",
        "options": ["Koffee", "Spice", "Elaine Thompson-Herah", "Lorna Goodison"],
        "topic": "Arts and culture",
        "year": "2020",
    },
    {
        "answer": "Chronixx",
        "title": "Chronixx",
        "question": "In Jamaica, which reggae artist was born Jamar Rolando McNaughton?",
        "options": ["Chronixx", "Popcaan", "Jimmy Cliff", "P. J. Patterson"],
        "topic": "Arts and culture",
        "year": "2017",
    },
    {
        "answer": "Vybz Kartel",
        "title": "Vybz Kartel",
        "question": "In Jamaica, which person is a Jamaican dancehall deejay also known as Worl Boss?",
        "options": ["Vybz Kartel", "Beenie Man", "Marcus Garvey", "Usain Bolt"],
        "topic": "Arts and culture",
        "year": "2016",
    },
    {
        "answer": "Beenie Man",
        "title": "Beenie Man",
        "question": "In Jamaica, which dancehall deejay is referred to as the King of Dancehall?",
        "options": ["Beenie Man", "Bounty Killer", "Bob Marley", "Michael Manley"],
        "topic": "Arts and culture",
        "year": "2000",
    },
    {
        "answer": "Bounty Killer",
        "title": "Bounty Killer",
        "question": "In Jamaica, which reggae and dancehall deejay founded the dancehall collective The Alliance?",
        "options": ["Bounty Killer", "Vybz Kartel", "Claude McKay", "Andrew Holness"],
        "topic": "Arts and culture",
        "year": "2005",
    },
    {
        "answer": "Popcaan",
        "title": "Popcaan",
        "question": "In Jamaica, which deejay, singer, and songwriter was formerly signed to Mixpak Records?",
        "options": ["Popcaan", "Chronixx", "Marlon James", "Merlene Ottey"],
        "topic": "Arts and culture",
        "year": "2014",
    },
    {
        "answer": "Damian Marley",
        "title": "Damian Marley",
        "question": "In Jamaica, which musician is the youngest son of reggae singer Bob Marley?",
        "options": ["Damian Marley", "Ziggy Marley", "Sean Paul", "Yohan Blake"],
        "topic": "Arts and culture",
        "year": "2005",
    },
    {
        "answer": "Ziggy Marley",
        "title": "Ziggy Marley",
        "question": "In Jamaica, which musician led the family band Ziggy Marley and the Melody Makers?",
        "options": ["Ziggy Marley", "Damian Marley", "Jimmy Cliff", "Marcus Garvey"],
        "topic": "Arts and culture",
        "year": "1988",
    },
    {
        "answer": "University of the West Indies at Mona",
        "title": "University of the West Indies at Mona",
        "question": "In Jamaica, which university campus is located in Mona, Jamaica?",
        "options": [
            "University of the West Indies at Mona",
            "Jamaica Stock Exchange",
            "Jamaica Observer",
            "Digicel",
        ],
        "topic": "Education",
        "year": "2015",
    },
    {
        "answer": "Jamaica Stock Exchange",
        "title": "Jamaica Stock Exchange",
        "question": "In Jamaica, which institution is the principal stock exchange of Jamaica?",
        "options": ["Jamaica Stock Exchange", "Digicel", "The Gleaner", "University of the West Indies at Mona"],
        "topic": "Business and economy",
        "year": "2018",
    },
    {
        "answer": "The Gleaner",
        "title": "The Gleaner",
        "question": "In Jamaica, which newspaper was founded in 1834 by Jacob and Joshua de Cordova?",
        "options": ["The Gleaner", "Jamaica Observer", "Digicel", "Jamaica Stock Exchange"],
        "topic": "Media",
        "year": "2014",
    },
    {
        "answer": "Jamaica Observer",
        "title": "Jamaica Observer",
        "question": "In Jamaica, which daily newspaper is based in Kingston, Jamaica?",
        "options": ["Jamaica Observer", "The Gleaner", "Jamaica Defence Force", "National Gallery of Jamaica"],
        "topic": "Media",
        "year": "2015",
    },
    {
        "answer": "Digicel",
        "title": "Digicel",
        "question": "In Jamaica, which telecommunications company is headquartered in Kingston, Jamaica?",
        "options": ["Digicel", "Jamaica Stock Exchange", "The Gleaner", "National Gallery of Jamaica"],
        "topic": "Business and economy",
        "year": "2016",
    },
    {
        "answer": "Jamaica Defence Force",
        "title": "Jamaica Defence Force",
        "question": "In Jamaica, which military organization is the combined military of Jamaica?",
        "options": ["Jamaica Defence Force", "Jamaica Constabulary Force", "Jamaica Stock Exchange", "The Gleaner"],
        "topic": "Public institutions",
        "year": "2015",
    },
    {
        "answer": "National Gallery of Jamaica",
        "title": "National Gallery of Jamaica",
        "question": "In Jamaica, which public art museum is located in Kingston, Jamaica?",
        "options": ["National Gallery of Jamaica", "Jamaica Observer", "Digicel", "Jamaica Defence Force"],
        "topic": "Arts and culture",
        "year": "2015",
    },
    {
        "answer": "Kingston",
        "title": "Kingston, Jamaica",
        "question": "In Jamaica, which city is the capital and largest city of Jamaica?",
        "options": ["Kingston", "Montego Bay", "Mandeville", "Ocho Rios"],
        "topic": "Places",
        "year": "2015",
    },
    {
        "answer": "Mandeville",
        "title": "Mandeville, Jamaica",
        "question": "In Jamaica, which town is the capital and largest town in the parish of Manchester?",
        "options": ["Mandeville", "Kingston", "Ocho Rios", "Montego Bay"],
        "topic": "Places",
        "year": "2015",
    },
    {
        "answer": "Ocho Rios",
        "title": "Ocho Rios",
        "question": "In Jamaica, which town is in the parish of Saint Ann on the north coast of Jamaica?",
        "options": ["Ocho Rios", "Mandeville", "Kingston", "Portmore"],
        "topic": "Places",
        "year": "2015",
    },
    {
        "answer": "People's National Party",
        "title": "People's National Party (Jamaica)",
        "question": "In Jamaica, which social democratic political party was founded in 1938 by Norman Manley?",
        "options": ["People's National Party", "Jamaica Labour Party", "Jamaica Defence Force", "Jamaica Stock Exchange"],
        "topic": "Politics",
        "year": "2015",
    },
    {
        "answer": "Jamaica Labour Party",
        "title": "Jamaica Labour Party",
        "question": "In Jamaica, which political party is one of the two major political parties alongside the People's National Party?",
        "options": ["Jamaica Labour Party", "People's National Party", "Jamaica Observer", "Jamaica Stock Exchange"],
        "topic": "Politics",
        "year": "2015",
    },
    {
        "answer": "Jamaica national football team",
        "title": "Jamaica national football team",
        "question": "In Jamaica, which national team represents Jamaica in men's international football?",
        "options": [
            "Jamaica national football team",
            "Jamaica national cricket team",
            "Jamaica Labour Party",
            "Jamaica Stock Exchange",
        ],
        "topic": "Sports",
        "year": "2015",
    },
    {
        "answer": "Jamaica national cricket team",
        "title": "Jamaica national cricket team",
        "question": "In Jamaica, which national team is the representative cricket team of Jamaica?",
        "options": [
            "Jamaica national cricket team",
            "Jamaica national football team",
            "Jamaica Defence Force",
            "Jamaica Observer",
        ],
        "topic": "Sports",
        "year": "2015",
    },
    {
        "answer": "Bob Marley Museum",
        "title": "Bob Marley Museum",
        "question": "In Jamaica, which tourist attraction in Kingston is dedicated to reggae musician Bob Marley?",
        "options": ["Bob Marley Museum", "Emancipation Park", "Jamaica Observer", "Jamaica Stock Exchange"],
        "topic": "Arts and culture",
        "year": "2015",
    },
    {
        "answer": "Norman Manley International Airport",
        "title": "Norman Manley International Airport",
        "question": "In Jamaica, which international airport serves Kingston, Jamaica?",
        "options": [
            "Norman Manley International Airport",
            "Sangster International Airport",
            "Bob Marley Museum",
            "Jamaica Stock Exchange",
        ],
        "topic": "Transport",
        "year": "2015",
    },
    {
        "answer": "Sangster International Airport",
        "title": "Sangster International Airport",
        "question": "In Jamaica, which international airport is located east of Montego Bay, Jamaica?",
        "options": [
            "Sangster International Airport",
            "Norman Manley International Airport",
            "Emancipation Park",
            "The Gleaner",
        ],
        "topic": "Transport",
        "year": "2015",
    },
    {
        "answer": "Emancipation Park",
        "title": "Emancipation Park (Kingston, Jamaica)",
        "question": "In Jamaica, which public park is located in Kingston, Jamaica?",
        "options": ["Emancipation Park", "Bob Marley Museum", "Jamaica Observer", "Jamaica Defence Force"],
        "topic": "Places",
        "year": "2015",
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
        writer.writerows(rows)


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


def norm(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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
            timeout=20,
            headers={"User-Agent": "LocalNewsQA-jamaica-curation/1.0"},
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
            url_title = quote(str(page.get("title", title) if page else title).replace(" ", "_"), safe="()_,")
            out[original_title] = {
                "url": f"https://en.wikipedia.org/wiki/{url_title}",
                "ok": bool(text),
                "status_code": "200" if text else "404",
                "final_url": page.get("fullurl", f"https://en.wikipedia.org/wiki/{url_title}") if page else "",
                "content_type": "application/json",
                "title": page.get("title", original_title) if page else original_title,
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


def find_support_sentence(seed: dict, fetch: dict, fetch_audit: Any) -> str:
    aliases = list(fetch_audit.answer_aliases(seed["answer"])) + [seed["answer"], seed["title"], fetch.get("title", "")]
    for sentence in split_sentences(fetch.get("text", ""))[:8]:
        sentence_norm = fetch_audit.normalize_text(sentence)
        has_answer = fetch_audit.contains_any(sentence_norm, aliases) or norm(fetch.get("title", "")) == norm(seed["answer"])
        has_country = fetch_audit.contains_any(sentence_norm, ["jamaica", "jamaican", "kingston"])
        if has_answer and has_country and 35 <= len(sentence) <= 450:
            if fetch_audit.contains_any(sentence_norm, aliases):
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


def build_row(seed: dict, idx: int, support: str, fetch: dict) -> dict:
    return {
        "id": "",
        "source_row_id": f"localnewsqa_explicit_jamaica_manual_curated_{idx:04d}",
        "country": "Jamaica",
        "continent": "North America",
        "target_country": "Jamaica",
        "contrast_country": "Canada",
        "topic": seed["topic"],
        "year": seed["year"],
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
        "source_split_type": "explicit_manual_curated",
        "localized_question": seed["question"],
        "curation_status": "manual_jamaica_wikipedia_supported",
        "curation_support_sentence": support,
    }


def main() -> None:
    explicit_audit = load_module(AUDIT_EXPLICIT_SCRIPT, "audit48")
    fetch_audit = load_module(FETCH_AUDIT_SCRIPT, "fetch32")
    builder = load_module(BUILDER_SCRIPT, "builder45")

    rows = read_jsonl(INPUT)
    ambiguous_ids = {row["source_row_id"] for row in read_jsonl(AMBIGUOUS)}
    fetches = explicit_audit.load_fetch_cache(FETCH_CACHE)
    page_fetches = fetch_wikipedia_pages([seed["title"] for seed in SEEDS])
    fetches.update({fetch["url"]: fetch for fetch in page_fetches.values() if fetch.get("ok")})

    used_sources = {row["source_row_id"] for row in rows}
    used_questions = {norm(row["question"]) for row in rows}
    counts = Counter(row["country"] for row in rows)
    need = 1000 - counts["Jamaica"]
    additions = []
    rejects = []

    for seed in SEEDS:
        if len(additions) >= need:
            break
        fetch = page_fetches.get(seed["title"], {})
        if not fetch.get("ok"):
            rejects.append({"answer": seed["answer"], "reason": "fetch_failed", "title": seed["title"]})
            continue
        source_id = f"localnewsqa_explicit_jamaica_manual_curated_{len(additions) + 1:04d}"
        if source_id in used_sources or norm(seed["question"]) in used_questions:
            rejects.append({"answer": seed["answer"], "reason": "duplicate_source_or_question", "title": seed["title"]})
            continue
        support = find_support_sentence(seed, fetch, fetch_audit)
        if not support:
            rejects.append({"answer": seed["answer"], "reason": "no_support_sentence", "title": seed["title"]})
            continue
        row = build_row(seed, len(additions) + 1, support, fetch)
        failures, warnings = explicit_audit.audit_explicit_row(row, fetches, ambiguous_ids, fetch_audit)
        if failures or warnings or high_conf_issue(row, fetches, builder) or anchor_issue(row, fetches, fetch_audit):
            rejects.append(
                {
                    "answer": seed["answer"],
                    "reason": "audit_reject",
                    "failures": " | ".join(failures),
                    "warnings": " | ".join(warnings),
                    "high_conf_issue": high_conf_issue(row, fetches, builder),
                    "anchor_issue": anchor_issue(row, fetches, fetch_audit),
                    "title": seed["title"],
                    "support": support,
                }
            )
            continue
        additions.append(row)
        used_sources.add(row["source_row_id"])
        used_questions.add(norm(row["question"]))

    if len(additions) != need:
        write_csv(REJECT_LOG, rejects)
        raise SystemExit(f"needed {need} Jamaica rows, accepted {len(additions)}")

    rows.extend(additions)
    rows.sort(key=lambda row: (COUNTRY_ORDER.index(row["country"]), norm(row.get("target_answer", "")), norm(row["question"])))
    for idx, row in enumerate(rows, start=1):
        row["id"] = f"localnewsqa_explicit_strict_defensible_1000_{idx:05d}"
        row["split_name"] = "LocalNewsQA-Explicit-Strict-Defensible-1000-Curated"

    source_ids = [row["source_row_id"] for row in rows]
    question_keys = [norm(row["question"]) for row in rows]
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
    write_dataset_csv(OUT_CSV, rows)
    write_csv(
        FILL_LOG,
        [
            {
                "source_row_id": row["source_row_id"],
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
        "source_split_type_counts": dict(Counter(row.get("source_split_type", "") for row in rows)),
        "manual_jamaica_rows_added": len(additions),
        "manual_jamaica_rejects": len(rejects),
        "duplicate_source_ids": len(source_ids) - len(set(source_ids)),
        "duplicate_questions": len(question_keys) - len(set(question_keys)),
        "ambiguous_overlap": len(set(source_ids) & ambiguous_ids),
        "valid": not validation_errors,
        "validation_errors": validation_errors,
        "paths": {
            "jsonl": str(OUT_JSONL),
            "csv": str(OUT_CSV),
            "fetch_cache": str(FETCH_CACHE),
            "manual_fill_log": str(FILL_LOG),
            "manual_reject_log": str(REJECT_LOG),
            "summary": str(SUMMARY),
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if validation_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
