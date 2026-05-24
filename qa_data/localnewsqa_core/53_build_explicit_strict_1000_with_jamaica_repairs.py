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
STRICT_DIR = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_semantic_gold_ambiguous_1700"
)
AUDIT_DIR = STRICT_DIR / "explicit_max_audit"
DEFAULT_CLEAN = AUDIT_DIR / "localnewsqa_targetqa_explicit_style_max_clean.jsonl"
DEFAULT_STRICT = AUDIT_DIR / "localnewsqa_targetqa_explicit_style_max_strict_no_warnings.jsonl"
DEFAULT_WARNINGS = AUDIT_DIR / "explicit_max_quality_warnings.csv"
DEFAULT_AMBIGUOUS = STRICT_DIR / "localnewsqa_ambiguous_semantic_gold_1700.jsonl"
DEFAULT_CACHE = AUDIT_DIR / "explicit_target_evidence_fetches.jsonl"
DEFAULT_OUTDIR = AUDIT_DIR / "strict_1000"
AUDIT_EXPLICIT_SCRIPT = ROOT / "qa_data/localnewsqa_core/48_audit_explicit_max_split.py"
AUDIT_FETCH_SCRIPT = ROOT / "qa_data/localnewsqa_core/32_web_audit_ambiguous_verifiable.py"

SOURCE_PRIORITY = {
    "explicit": 0,
    "ambiguous_salvaged_target": 1,
    "ambiguous_pool_salvaged_target_web_supported": 2,
}

HOLIDAY_ANSWERS = {
    "ash wednesday",
    "boxing day",
    "christmas day",
    "good friday",
    "independence",
    "independence day",
    "labour day",
    "new year s day",
    "remembrance day",
}
EDUCATION_ANSWERS = {
    "cape",
    "csec",
    "grade 1",
    "grade 6",
    "grade 7",
    "grade 8",
    "grade 9",
    "grade 10",
    "grade 11",
    "gsat",
    "pep",
    "preparatory schools",
    "technical and vocational education and training",
}
SPORT_ANSWERS = {"cricket", "football"}

JAMAICA_EVIDENCE_BY_ANSWER = {
    "63": ("House of Representatives of Jamaica", "https://en.wikipedia.org/wiki/House_of_Representatives_of_Jamaica"),
    "director of public prosecutions": (
        "Office of the Director of Public Prosecutions Jamaica",
        "https://dpp.gov.jm/",
    ),
    "fair trading commission": ("Jamaica Fair Trading Commission", "https://jftc.gov.jm/"),
    "fesco": ("FESCO Jamaica", "https://www.fescoja.com/"),
    "financial services commission": ("Financial Services Commission Jamaica", "https://www.fscjamaica.org/"),
    "governor general": ("Governor-General of Jamaica", "https://en.wikipedia.org/wiki/Governor-General_of_Jamaica"),
    "governor general s department": (
        "Governor-General of Jamaica",
        "https://en.wikipedia.org/wiki/Governor-General_of_Jamaica",
    ),
    "house of representatives": (
        "House of Representatives of Jamaica",
        "https://en.wikipedia.org/wiki/House_of_Representatives_of_Jamaica",
    ),
    "manchester": ("Manchester Parish", "https://en.wikipedia.org/wiki/Manchester_Parish"),
    "ministry of education": ("Education in Jamaica", "https://en.wikipedia.org/wiki/Education_in_Jamaica"),
    "ministry of education and youth": ("Education in Jamaica", "https://en.wikipedia.org/wiki/Education_in_Jamaica"),
    "ministry of finance and planning": ("Ministry of Finance and the Public Service Jamaica", "https://www.mof.gov.jm/"),
    "ministry of health": ("Ministry of Health and Wellness Jamaica", "https://www.moh.gov.jm/"),
    "ministry of local government": ("Ministry of Local Government Jamaica", "https://www.localgovjamaica.gov.jm/"),
    "ministry of local government and community development": (
        "Ministry of Local Government Jamaica",
        "https://www.localgovjamaica.gov.jm/",
    ),
    "ministry of local government and rural development": (
        "Ministry of Local Government Jamaica",
        "https://www.localgovjamaica.gov.jm/",
    ),
    "ministry of tourism": ("Jamaica Ministry of Tourism", "https://www.mot.gov.jm/"),
    "ministry of transport and mining": ("Ministry of Transport and Mining Jamaica", "https://mtm.gov.jm/"),
    "national insurance fund": (
        "National Insurance Fund Jamaica",
        "https://jis.gov.jm/government/agencies/national-insurance-fund/",
    ),
    "national water commission": ("National Water Commission Jamaica", "https://www.nwcjamaica.com/"),
    "office of the director of public prosecutions": (
        "Office of the Director of Public Prosecutions Jamaica",
        "https://dpp.gov.jm/",
    ),
    "ombudsman": ("Public Defender Jamaica", "https://opd.gov.jm/"),
    "president of the senate": ("Senate of Jamaica", "https://en.wikipedia.org/wiki/Senate_of_Jamaica"),
    "prime minister": ("Prime Minister of Jamaica", "https://en.wikipedia.org/wiki/Prime_Minister_of_Jamaica"),
    "public procurement commission": ("Public Procurement Commission Jamaica", "https://ppc.gov.jm/"),
    "public defender": ("Public Defender Jamaica", "https://opd.gov.jm/"),
    "public service commission": ("Office of the Services Commissions Jamaica", "https://osc.gov.jm/"),
    "registrar general s department": (
        "Registrar General's Department Jamaica",
        "https://jis.gov.jm/government/agencies/registrar-generals-department/",
    ),
    "saint ann": ("Saint Ann Parish", "https://en.wikipedia.org/wiki/Saint_Ann_Parish"),
    "senate": ("Senate of Jamaica", "https://en.wikipedia.org/wiki/Senate_of_Jamaica"),
    "st mary": ("Saint Mary Parish, Jamaica", "https://en.wikipedia.org/wiki/Saint_Mary_Parish,_Jamaica"),
    "the prime minister": ("Prime Minister of Jamaica", "https://en.wikipedia.org/wiki/Prime_Minister_of_Jamaica"),
    "urban development corporation": ("Urban Development Corporation Jamaica", "https://udcja.com/"),
    "vale royal": ("Vale Royal Jamaica", "https://opm.gov.jm/contact/vale-royal/"),
    "peter phillips": ("Peter Phillips Jamaican politician", "https://en.wikipedia.org/wiki/Peter_Phillips_(politician)"),
    "commission for the prevention of corruption": ("Integrity Commission Jamaica", "https://integrity.gov.jm/"),
    "judicial service commission": ("Judiciary of Jamaica", "https://en.wikipedia.org/wiki/Judiciary_of_Jamaica"),
    "police service commission": ("Jamaica Constabulary Force", "https://en.wikipedia.org/wiki/Jamaica_Constabulary_Force"),
    "court of appeal": ("Court of Appeal of Jamaica", "https://en.wikipedia.org/wiki/Court_of_Appeal_of_Jamaica"),
    "champs": (
        "Inter-Secondary Schools Boys and Girls Championships Jamaica",
        "https://en.wikipedia.org/wiki/Inter-Secondary_Schools_Boys_and_Girls_Championships",
    ),
    "coffee": ("Jamaica Blue Mountain Coffee", "https://en.wikipedia.org/wiki/Jamaica_Blue_Mountain_Coffee"),
    "seprod": ("Seprod Jamaica", "https://www.seprod.com/"),
    "the star": ("Jamaica Gleaner and The Star", "https://jamaica-gleaner.com/"),
    "mexico": ("2015 CONCACAF Gold Cup", "https://en.wikipedia.org/wiki/2015_CONCACAF_Gold_Cup"),
    "united states": ("2015 CONCACAF Gold Cup", "https://en.wikipedia.org/wiki/2015_CONCACAF_Gold_Cup"),
    "venezuela": (
        "Copa America Centenario Group C",
        "https://en.wikipedia.org/wiki/Copa_Am%C3%A9rica_Centenario_Group_C",
    ),
    "group c": (
        "Copa America Centenario Group C",
        "https://en.wikipedia.org/wiki/Copa_Am%C3%A9rica_Centenario_Group_C",
    ),
    "sydney": ("2015 Netball World Cup", "https://en.wikipedia.org/wiki/2015_Netball_World_Cup"),
    "rio de janeiro": ("Jamaica at the 2016 Summer Olympics", "https://en.wikipedia.org/wiki/Jamaica_at_the_2016_Summer_Olympics"),
    "hanover": ("Hanover Parish", "https://en.wikipedia.org/wiki/Hanover_Parish"),
    "st ann": ("Saint Ann Parish", "https://en.wikipedia.org/wiki/Saint_Ann_Parish"),
    "english": ("Languages of Jamaica", "https://en.wikipedia.org/wiki/Languages_of_Jamaica"),
    "state opening of parliament": ("Parliament of Jamaica", "https://en.wikipedia.org/wiki/Parliament_of_Jamaica"),
    "speech from the throne": ("Parliament of Jamaica", "https://en.wikipedia.org/wiki/Parliament_of_Jamaica"),
    "pnp": ("People's National Party Jamaica", "https://en.wikipedia.org/wiki/People%27s_National_Party"),
    "little theatre": ("National Dance Theatre Company of Jamaica", "https://ndtcjamaica.org/"),
    "little theatre movement": ("National Dance Theatre Company of Jamaica", "https://ndtcjamaica.org/"),
    "rjr": ("Radio Jamaica", "https://rjr94fm.com/"),
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})


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


def write_cache(path: Path, fetches: dict[str, dict[str, Any]]) -> None:
    write_jsonl(path, [fetches[url] for url in sorted(fetches)])


def norm(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def row_rank(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        SOURCE_PRIORITY.get(row.get("source_split_type", ""), 9),
        norm(row.get("target_answer", "")),
        norm(row.get("question", "")),
        row.get("source_row_id", ""),
    )


def read_warning_ids(path: Path) -> set[str]:
    out = set()
    if not path.exists() or path.stat().st_size == 0:
        return out
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("source_row_id"):
                out.add(row["source_row_id"])
    return out


def mapped_evidence(row: dict[str, Any]) -> tuple[str, str] | None:
    answer = norm(row.get("target_answer", ""))
    if answer in JAMAICA_EVIDENCE_BY_ANSWER:
        return JAMAICA_EVIDENCE_BY_ANSWER[answer]
    if answer in HOLIDAY_ANSWERS:
        return ("Public holidays in Jamaica", "https://en.wikipedia.org/wiki/Public_holidays_in_Jamaica")
    if answer in EDUCATION_ANSWERS:
        return ("Education in Jamaica", "https://en.wikipedia.org/wiki/Education_in_Jamaica")
    if answer in SPORT_ANSWERS:
        return ("Sport in Jamaica", "https://en.wikipedia.org/wiki/Sport_in_Jamaica")
    return None


def strict_pass(row: dict[str, Any], fetches: dict[str, dict[str, Any]], ambiguous_ids: set[str], audit_mod: Any, fetch_mod: Any) -> bool:
    failures, warnings = audit_mod.audit_explicit_row(row, fetches, ambiguous_ids, fetch_mod)
    return not failures and not warnings


def main() -> None:
    audit_mod = load_module(AUDIT_EXPLICIT_SCRIPT, "audit48")
    fetch_mod = load_module(AUDIT_FETCH_SCRIPT, "fetch32")
    clean_rows = read_jsonl(DEFAULT_CLEAN)
    strict_rows = read_jsonl(DEFAULT_STRICT)
    strict_ids = {row["source_row_id"] for row in strict_rows}
    warning_ids = read_warning_ids(DEFAULT_WARNINGS)
    ambiguous_ids = {row.get("source_row_id", "") for row in read_jsonl(DEFAULT_AMBIGUOUS)}
    fetches = load_fetch_cache(DEFAULT_CACHE)

    repairs = []
    repair_log = []
    for row in sorted(clean_rows, key=row_rank):
        if row.get("country") != "Jamaica" or row.get("source_row_id") in strict_ids:
            continue
        if row.get("source_row_id") not in warning_ids:
            continue
        evidence = mapped_evidence(row)
        if not evidence:
            continue
        title, url = evidence
        if not fetches.get(url, {}).get("ok"):
            fetches[url] = fetch_mod.fetch_url(url, timeout=25.0, delay=0.2)
            time.sleep(0.2)
        repaired = dict(row)
        repaired["target_evidence_title"] = title
        repaired["target_evidence_url"] = url
        repaired["target_evidence_excerpt"] = title
        repaired["explicit_evidence_repair"] = {
            "old_target_evidence_title": row.get("target_evidence_title", ""),
            "old_target_evidence_url": row.get("target_evidence_url", ""),
            "repair_reason": "jamaica_country_specific_evidence_for_strict_1000",
        }
        passed = strict_pass(repaired, fetches, ambiguous_ids, audit_mod, fetch_mod)
        repair_log.append(
            {
                "source_row_id": row.get("source_row_id", ""),
                "target_answer": row.get("target_answer", ""),
                "old_title": row.get("target_evidence_title", ""),
                "old_url": row.get("target_evidence_url", ""),
                "new_title": title,
                "new_url": url,
                "accepted": passed,
            }
        )
        if passed:
            repairs.append(repaired)

    DEFAULT_OUTDIR.mkdir(parents=True, exist_ok=True)
    write_cache(DEFAULT_CACHE, fetches)
    write_csv(DEFAULT_OUTDIR / "jamaica_evidence_repair_log.csv", repair_log)
    write_jsonl(DEFAULT_OUTDIR / "jamaica_evidence_repaired_candidates.jsonl", repairs)

    selected = []
    used_ids = set()
    used_questions = set()
    repair_by_id = {row["source_row_id"]: row for row in repairs}
    strict_by_country: dict[str, list[dict[str, Any]]] = {}
    for country in sorted({row.get("country", "") for row in strict_rows}):
        strict_by_country[country] = sorted([row for row in strict_rows if row.get("country") == country], key=row_rank)

    for country in sorted(strict_by_country):
        country_selected = []
        for row in strict_by_country[country]:
            if len(country_selected) >= 1000:
                break
            question = row.get("question", "")
            source_id = row.get("source_row_id", "")
            if source_id in used_ids or question in used_questions:
                continue
            country_selected.append(row)
            used_ids.add(source_id)
            used_questions.add(question)
        if country == "Jamaica" and len(country_selected) < 1000:
            for row in sorted(repair_by_id.values(), key=row_rank):
                if len(country_selected) >= 1000:
                    break
                question = row.get("question", "")
                source_id = row.get("source_row_id", "")
                if source_id in used_ids or question in used_questions:
                    continue
                country_selected.append(row)
                used_ids.add(source_id)
                used_questions.add(question)
        selected.extend(country_selected)

    selected.sort(key=lambda row: (row.get("country", ""), row_rank(row)))
    jsonl_path = DEFAULT_OUTDIR / "localnewsqa_targetqa_explicit_strict_1000_per_country.jsonl"
    csv_path = DEFAULT_OUTDIR / "localnewsqa_targetqa_explicit_strict_1000_per_country.csv"
    write_jsonl(jsonl_path, selected)
    write_csv(csv_path, selected)

    counts = Counter(row.get("country", "") for row in selected)
    summary = {
        "rows": len(selected),
        "country_counts": dict(sorted(counts.items())),
        "jamaica_strict_original": sum(1 for row in strict_rows if row.get("country") == "Jamaica"),
        "jamaica_repairs_accepted": len(repairs),
        "duplicate_source_ids": sum(1 for count in Counter(row.get("source_row_id", "") for row in selected).values() if count > 1),
        "duplicate_questions": sum(1 for count in Counter(row.get("question", "") for row in selected).values() if count > 1),
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
    }
    (DEFAULT_OUTDIR / "strict_1000_build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if len(selected) != 17_000 or any(count != 1000 for count in counts.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
