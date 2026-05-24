#!/usr/bin/env python3

import csv
import importlib.util
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_semantic_gold_ambiguous_1700"
)
AUDIT_DIR = BASE_DIR / "explicit_max_audit"
STRICT_DIR = AUDIT_DIR / "strict_1000"
SPLIT_JSONL = STRICT_DIR / "localnewsqa_targetqa_explicit_strict_1000_per_country_polished.jsonl"
SPLIT_CSV = STRICT_DIR / "localnewsqa_targetqa_explicit_strict_1000_per_country_polished.csv"
FETCH_CACHE = AUDIT_DIR / "explicit_target_evidence_fetches.jsonl"
OUT_DIR = STRICT_DIR / "strict_relation_probe_repairs"
FETCH_SCRIPT = ROOT / "qa_data/localnewsqa_core/32_web_audit_ambiguous_verifiable.py"


Repair = dict[str, str]

REPAIRS: dict[str, Repair] = {
    "localnewsqa_explicit_0001309": {
        "question": "In Canada, which independent electoral boundaries commissions review federal electoral district boundaries after each decennial census?",
        "title": "Process Assessment Report: 2012 Redistribution of Federal Electoral Districts",
        "url": "https://www.elections.ca/content.aspx?dir=rep%2Foth%2Febrar&document=p3&lang=e&section=res",
        "excerpt": "Federal Electoral Boundaries Commissions: ten independent electoral boundaries commissions revise districts after a decennial census.",
    },
    "localnewsqa_explicit_0002007": {
        "question": "Which pair of Canadian cities hosted the 2015 World Junior Ice Hockey Championship?",
        "title": "2015 World Junior Ice Hockey Championships",
        "url": "https://en.wikipedia.org/wiki/2015_World_Junior_Ice_Hockey_Championships",
        "excerpt": "2015 World Junior Ice Hockey Championship host country Canada; host cities Toronto and Montreal.",
    },
    "localnewsqa_explicit_0015159": {
        "question": "Which Ghanaian department registers companies and business names?",
        "title": "Registrar General's Department",
        "url": "https://en.wikipedia.org/wiki/Registrar_General%27s_Department",
        "excerpt": "Registrar-General's Department is Ghana's agency responsible for registration of companies and businesses.",
    },
    "localnewsqa_explicit_0015353": {
        "question": "In Ghana, what was the usual duration of senior high school after it reverted from the temporary four-year structure?",
        "title": "Education in Ghana",
        "url": "https://en.wikipedia.org/wiki/Education_in_Ghana",
        "excerpt": "Education in Ghana: the senior high school curriculum reverted to three years after a four-year structure.",
    },
    "localnewsqa_explicit_0015328": {
        "question": "What is the usual duration of junior high school in Ghana?",
        "title": "Education in Ghana",
        "url": "https://en.wikipedia.org/wiki/Education_in_Ghana",
        "excerpt": "Education in Ghana: junior high school lasts three years.",
    },
    "localnewsqa_explicit_0015327": {
        "question": "What is the usual duration of senior high school in Ghana?",
        "title": "Education in Ghana",
        "url": "https://en.wikipedia.org/wiki/Education_in_Ghana",
        "excerpt": "Education in Ghana: the usual senior high school curriculum lasts three years.",
    },
    "localnewsqa_explicit_0008556": {
        "question": "In 2024, Hong Kong authorities said funded taught undergraduate programmes would have a non-local student quota of what share?",
        "title": "LCQ14: Non-local student quota for funded taught undergraduate programmes",
        "url": "https://www.info.gov.hk/gia/general/202312/06/P2023120600243p.htm",
        "excerpt": "Hong Kong non-local student quota for funded taught undergraduate programmes was raised to 40 per cent from 2024/25.",
    },
    "localnewsqa_explicit_0008810": {
        "question": "Which club finished third in Group B of the 2018-19 Hong Kong Sapling Cup?",
        "title": "2018-19 Hong Kong Sapling Cup",
        "url": "https://en.wikipedia.org/wiki/2018%E2%80%9319_Hong_Kong_Sapling_Cup",
        "excerpt": "2018-19 Hong Kong Sapling Cup Group B table: Southern finished third.",
    },
    "localnewsqa_explicit_0004563": {
        "question": "In India, the Konkan is a western coastal plain lying by which sea?",
        "title": "Konkan",
        "url": "https://www.britannica.com/place/Konkan",
        "excerpt": "Konkan is a coastal plain of western India lying by the Arabian Sea; answer: Arabian Sea coast.",
    },
    "localnewsqa_explicit_0018394": {
        "question": "In Ireland, how many seats were in Dail Eireann at the 2020 general election?",
        "title": "2020 Irish general election",
        "url": "https://en.wikipedia.org/wiki/2020_Irish_general_election",
        "excerpt": "The 2020 Irish general election had 160 seats in Dail Eireann.",
    },
    "localnewsqa_explicit_0019417": {
        "question": "In Ireland, what is the name of Dublin's road tunnel under the River Liffey that forms part of the M50?",
        "title": "Dublin Port Tunnel",
        "url": "https://en.wikipedia.org/wiki/Dublin_Port_Tunnel",
        "excerpt": "Dublin Port Tunnel is a road tunnel in Dublin, Ireland, under the River Liffey and part of the M50.",
    },
    "localnewsqa_explicit_0002742": {
        "question": "Which Jamaican agency is responsible for assessment and certification of the Jamaican workforce?",
        "title": "HEART Trust NTA",
        "url": "https://www.jsif.org/content/heart-trust-nta",
        "excerpt": "HEART Trust/NTA is responsible for assessment and certification of the Jamaican workforce.",
    },
    "localnewsqa_explicit_0002809": {
        "question": "Which Jamaican agency is linked to technical and vocational certification through the NVQ-J qualification?",
        "title": "JCTE, HEART-Trust / NTA launch new Occupational Studies degree project",
        "url": "https://moey.gov.jm/jcte-heart-trust-nta-launch-new-occupational-studies-degree-project/",
        "excerpt": "HEART Trust/NTA and the National Vocational Qualification of Jamaica (NVQ-J) are linked in Jamaican vocational certification.",
    },
    "localnewsqa_explicit_0002765": {
        "question": "Which Jamaican training organization is commonly known as HEART and provides vocational workforce training?",
        "title": "HEART Trust NTA",
        "url": "https://www.jsif.org/content/heart-trust-nta",
        "excerpt": "HEART Trust/NTA builds a trained, qualified Jamaican workforce through skills and vocational training.",
    },
    "localnewsqa_explicit_0002403": {
        "question": "Which ministry sought to make Portmore Jamaica's 15th parish by tabling a motion in the House of Representatives?",
        "title": "Jamaican Government Planning to Make Portmore the Island's 15th Parish",
        "url": "https://www.caribbeannationalweekly.com/caribbean-breaking-news-featured/jamaican-government-planning-to-make-portmore-the-islands-15th-parish/",
        "excerpt": "Jamaica's Ministry of Local Government and Rural Development was to table a motion to make Portmore the island's 15th parish.",
    },
    "localnewsqa_explicit_0003411": {
        "question": "In Jamaica, what is the name of the large seaport in Kingston tied to transshipment hub reporting?",
        "title": "Port of Kingston",
        "url": "https://mfaft.gov.jm/site/port-of-kingston/",
        "excerpt": "Port of Kingston is Jamaica's Kingston seaport and transshipment hub.",
    },
    "localnewsqa_explicit_0003339": {
        "question": "What is the name of the Jamaican port near Kingston central to logistics and transshipment reporting?",
        "title": "Port of Kingston",
        "url": "https://mfaft.gov.jm/site/port-of-kingston/",
        "excerpt": "Port of Kingston is the Jamaican port near Kingston central to logistics and transshipment.",
    },
    "localnewsqa_explicit_0002638": {
        "question": "Which Jamaican port is a main container transshipment hub?",
        "title": "Port Authority of Jamaica",
        "url": "https://en.wikipedia.org/wiki/Port_Authority_of_Jamaica",
        "excerpt": "Port of Kingston is a Jamaica container/transshipment hub under the Port Authority of Jamaica.",
    },
    "localnewsqa_explicit_0003397": {
        "question": "The Jamaica Urban Transit Company has a major Half Way Tree hub in which neighboring parish of Kingston?",
        "title": "Jamaica Urban Transit Company",
        "url": "https://en.wikipedia.org/wiki/Jamaica_Urban_Transit_Company",
        "excerpt": "Jamaica Urban Transit Company lists Half Way Tree Transport Centre in Saint Andrew as a primary hub.",
    },
    "localnewsqa_explicit_0003402": {
        "question": "The Mona Reservoir is located in which Jamaican parish?",
        "title": "Mona, Jamaica",
        "url": "https://en.wikipedia.org/wiki/Mona,_Jamaica",
        "excerpt": "Mona is in Saint Andrew Parish, Jamaica, and is the site of Mona Reservoir.",
    },
    "localnewsqa_explicit_0002511": {
        "question": "Which Jamaican institution is the Mona campus of The University of the West Indies?",
        "title": "University of the West Indies",
        "url": "https://en.wikipedia.org/wiki/University_of_the_West_Indies_at_Mona",
        "excerpt": "The University of the West Indies, Mona is the Jamaican Mona campus of the University of the West Indies.",
    },
    "localnewsqa_explicit_0002837": {
        "question": "Which Jamaican university is the Mona flagship constituent unit of The University of the West Indies?",
        "title": "University of the West Indies",
        "url": "https://en.wikipedia.org/wiki/University_of_the_West_Indies_at_Mona",
        "excerpt": "The University of the West Indies, Mona is the flagship constituent unit in Mona, Jamaica.",
    },
    "localnewsqa_explicit_0014840": {
        "question": "In Kenya, which solar power plant in Garissa County was expected to be the largest in East and Central Africa?",
        "title": "Garissa Solar Power Station",
        "url": "https://en.wikipedia.org/wiki/Garissa_Solar_Power_Station",
        "excerpt": "Garissa Solar Power Plant in Kenya's Garissa County was expected to be the largest in East and Central Africa.",
    },
    "localnewsqa_explicit_0013949": {
        "question": "Which Kenyan authority oversees transport planning and management within the Nairobi Metropolitan Area?",
        "title": "The Nairobi Metropolitan Area Transport Authority Order",
        "url": "https://new.kenyalaw.org/akn/ke/act/ln/2017/18",
        "excerpt": "Kenya Law: Nairobi Metropolitan Area Transport Authority oversees public transport planning and mobility in the Nairobi Metropolitan Area.",
    },
    "localnewsqa_explicit_0009457": {
        "question": "Which Malaysian statutory body is abbreviated ECERDC in the East Coast Economic Region?",
        "title": "East Coast Economic Region",
        "url": "https://en.wikipedia.org/wiki/East_Coast_Economic_Region",
        "excerpt": "East Coast Economic Region Development Council (ECERDC) is the Malaysian statutory body for the East Coast Economic Region.",
    },
    "localnewsqa_explicit_0011906": {
        "question": "Under Nigeria's 6-3-3-4 education system, how many years are represented by the final '4' in tertiary education?",
        "title": "Education in Nigeria",
        "url": "https://en.wikipedia.org/wiki/Education_in_Nigeria",
        "excerpt": "Education in Nigeria: the 6-3-3-4 system includes four years in a tertiary institution; answer: 4 years.",
    },
    "localnewsqa_explicit_0011582": {
        "question": "Nigeria's president is elected for a term of how many years?",
        "title": "President of Nigeria",
        "url": "https://en.wikipedia.org/wiki/President_of_Nigeria",
        "excerpt": "President of Nigeria term length is four years.",
    },
    "localnewsqa_explicit_0004954": {
        "question": "Which Pakistani export sector is centered on textiles and apparel?",
        "title": "Textile industry in Pakistan",
        "url": "https://en.wikipedia.org/wiki/Textile_industry_in_Pakistan",
        "excerpt": "Pakistan's textile industry comprises textiles and apparel and is a leading export sector.",
    },
    "localnewsqa_explicit_0007367": {
        "question": "Which language pair is used as primary languages in Sri Lanka's education system?",
        "title": "Education in Sri Lanka",
        "url": "https://en.wikipedia.org/wiki/Education_in_Sri_Lanka",
        "excerpt": "Education in Sri Lanka lists Sinhala and Tamil among the primary languages and school media.",
    },
    "localnewsqa_explicit_0007889": {
        "question": "Which language pair appears in the English name 'Sinhala and Tamil New Year' in Sri Lanka?",
        "title": "Sinhala and Tamil New Year",
        "url": "https://en.wikipedia.org/wiki/Sinhala_and_Tamil_New_Year",
        "excerpt": "Sinhala and Tamil New Year is the standard English name of the Sri Lankan April New Year holiday.",
    },
    "localnewsqa_explicit_0016258": {
        "question": "Which Tanzanian body is the governing authority of Dar es Salaam at the city council level?",
        "title": "Mayor of Dar es Salaam",
        "url": "https://en.wikipedia.org/wiki/Mayor_of_Dar_es_Salaam",
        "excerpt": "Dar es Salaam City Council is the governing authority of Dar es Salaam, Tanzania.",
    },
    "localnewsqa_explicit_0016632": {
        "question": "Which Tanzanian media company publishes The Guardian and Nipashe?",
        "title": "IPP Media Limited",
        "url": "https://tanzania.mom-gmr.org/en/owners/companies/detail/company//ipp-media-limited-2/",
        "excerpt": "IPP Media publishes The Guardian and Nipashe in Tanzania.",
    },
    "localnewsqa_explicit_0016701": {
        "question": "Which Tanzanian media group includes the newspaper The Guardian?",
        "title": "IPP Media Limited",
        "url": "https://tanzania.mom-gmr.org/en/owners/companies/detail/company//ipp-media-limited-2/",
        "excerpt": "IPP Media is a Tanzanian media group whose newspapers include The Guardian.",
    },
    "localnewsqa_explicit_0016716": {
        "question": "Which Tanzanian media owner has newspapers, television stations, and radio stations under the IPP brand?",
        "title": "IPP Media Limited",
        "url": "https://tanzania.mom-gmr.org/en/owners/companies/detail/company//ipp-media-limited-2/",
        "excerpt": "IPP Media has newspapers, television stations, and radio stations in Tanzania.",
    },
}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
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
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def load_fetch_cache(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("url"):
                    out[row["url"]] = row
    return out


def write_fetch_cache(path: Path, fetches: dict[str, dict[str, Any]]) -> None:
    write_jsonl(path, [fetches[url] for url in sorted(fetches)])


def norm(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def apply_repairs(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repaired = []
    logs = []
    for row in rows:
        working = dict(row)
        source_id = working.get("source_row_id", "")
        repair = REPAIRS.get(source_id)
        if repair:
            old = {
                "old_question": working.get("question", ""),
                "old_original_question": working.get("original_question", ""),
                "old_localized_question": working.get("localized_question", ""),
                "old_target_evidence_title": working.get("target_evidence_title", ""),
                "old_target_evidence_url": working.get("target_evidence_url", ""),
                "old_target_evidence_excerpt": working.get("target_evidence_excerpt", ""),
            }
            working["question"] = repair["question"]
            working["original_question"] = repair["question"]
            working["localized_question"] = repair["question"]
            working["target_evidence_title"] = repair["title"]
            working["target_evidence_url"] = repair["url"]
            working["target_evidence_excerpt"] = repair["excerpt"]
            working["evidence_hint"] = repair["excerpt"]
            logs.append(
                {
                    "source_row_id": source_id,
                    "country": working.get("country", ""),
                    "target_answer": working.get("target_answer", ""),
                    **old,
                    "new_question": working.get("question", ""),
                    "new_target_evidence_title": working.get("target_evidence_title", ""),
                    "new_target_evidence_url": working.get("target_evidence_url", ""),
                    "new_target_evidence_excerpt": working.get("target_evidence_excerpt", ""),
                }
            )
        repaired.append(working)
    return repaired, logs


def fetch_needed(urls: set[str], fetches: dict[str, dict[str, Any]], fetch_mod: Any) -> None:
    for url in sorted(urls):
        if fetches.get(url, {}).get("ok"):
            continue
        fetches[url] = fetch_mod.fetch_url(url, timeout=25.0, delay=0.2)
        time.sleep(0.2)


def option_integrity(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    failures = []
    for row in rows:
        options = row.get("options") or []
        normalized = [norm(option) for option in options]
        if len(options) != 4 or len(normalized) != len(set(normalized)) or normalized.count(norm(row.get("target_answer", ""))) != 1:
            failures.append(
                {
                    "source_row_id": row.get("source_row_id", ""),
                    "question": row.get("question", ""),
                    "target_answer": row.get("target_answer", ""),
                    "issue": "option_shape_or_target_membership",
                }
            )
    return failures


def main() -> None:
    fetch_mod = load_module(FETCH_SCRIPT, "fetch32")
    rows = read_jsonl(SPLIT_JSONL)
    repaired, logs = apply_repairs(rows)
    fetches = load_fetch_cache(FETCH_CACHE)
    fetch_needed({repair["url"] for repair in REPAIRS.values()}, fetches, fetch_mod)
    write_fetch_cache(FETCH_CACHE, fetches)
    write_jsonl(SPLIT_JSONL, repaired)
    write_csv(SPLIT_CSV, repaired)

    fetch_failures = [
        {
            "url": url,
            "title": repair["title"],
            "error": fetches.get(url, {}).get("error", "missing_fetch"),
        }
        for url, repair in {repair["url"]: repair for repair in REPAIRS.values()}.items()
        if not fetches.get(url, {}).get("ok")
    ]
    option_failures = option_integrity([row for row in repaired if row.get("source_row_id") in REPAIRS])
    summary = {
        "input_jsonl": str(SPLIT_JSONL),
        "output_jsonl": str(SPLIT_JSONL),
        "output_csv": str(SPLIT_CSV),
        "rows_total": len(repaired),
        "repairs": len(logs),
        "unique_repair_urls": len({repair["url"] for repair in REPAIRS.values()}),
        "repair_fetch_ok": sum(1 for repair in REPAIRS.values() if fetches.get(repair["url"], {}).get("ok")),
        "fetch_failures": len(fetch_failures),
        "option_failures": len(option_failures),
        "duplicate_source_ids": sum(
            1 for count in Counter(row.get("source_row_id", "") for row in repaired).values() if count > 1
        ),
        "duplicate_questions": sum(1 for count in Counter(row.get("question", "") for row in repaired).values() if count > 1),
        "by_country": dict(Counter(row.get("country", "") for row in repaired if row.get("source_row_id") in REPAIRS)),
        "outputs": {
            "repair_log": str(OUT_DIR / "repair_log.csv"),
            "fetch_failures": str(OUT_DIR / "fetch_failures.csv"),
            "option_failures": str(OUT_DIR / "option_failures.csv"),
            "summary": str(OUT_DIR / "summary.json"),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "repair_log.csv", logs)
    write_csv(OUT_DIR / "fetch_failures.csv", fetch_failures)
    write_csv(OUT_DIR / "option_failures.csv", option_failures)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if fetch_failures or option_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
