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
STRICT_DIR = BASE / "explicit_max_audit/strict_defensible_954"
INPUT_JSONL = STRICT_DIR / "localnewsqa_targetqa_explicit_strict_defensible_954_per_country.jsonl"
INPUT_CSV = STRICT_DIR / "localnewsqa_targetqa_explicit_strict_defensible_954_per_country.csv"
FETCH_CACHE = STRICT_DIR / "strict_defensible_target_evidence_fetches.jsonl"
REPAIR_LOG = STRICT_DIR / "answer_anchor_manual_evidence_repair_log.csv"
SUMMARY = STRICT_DIR / "answer_anchor_manual_evidence_repair_summary.json"
AMBIGUOUS = BASE / "localnewsqa_ambiguous_semantic_gold_1700.jsonl"

AUDIT_EXPLICIT_SCRIPT = ROOT / "qa_data/localnewsqa_core/48_audit_explicit_max_split.py"
FETCH_AUDIT_SCRIPT = ROOT / "qa_data/localnewsqa_core/32_web_audit_ambiguous_verifiable.py"
BUILDER_SCRIPT = ROOT / "qa_data/localnewsqa_core/45_build_relation_strict_gold_ambiguous.py"

SEED_FETCH_CACHES = [
    BASE / "semantic_gold_selected_evidence_fetches.jsonl",
    BASE / "explicit_max_audit/explicit_target_evidence_fetches.jsonl",
    BASE / "explicit_max_audit/strict_1000/polished_audit/explicit_target_evidence_fetches.jsonl",
    STRICT_DIR / "audit/explicit_target_evidence_fetches.jsonl",
    FETCH_CACHE,
]

REPAIRS = {
    "localnewsqa_explicit_0004251": {
        "title": "2016 BRICS U-17 Football Cup",
        "url": "https://en.wikipedia.org/wiki/2016_BRICS_U-17_Football_Cup",
        "excerpt": "The 2016 BRICS U-17 Football Cup was held in Goa, India. Venue listings for the tournament include Athletic Stadium, Bambolim, as a match venue.",
    },
    "localnewsqa_explicit_0018395": {
        "title": "Seanad Éireann",
        "url": "https://en.wikipedia.org/wiki/Seanad_%C3%89ireann",
        "excerpt": "Seanad Éireann is the upper house of the Oireachtas, the legislature of Ireland. It consists of 60 senators.",
    },
    "localnewsqa_ambig_0035064": {
        "title": "Attorney General of Ireland",
        "url": "https://en.wikipedia.org/wiki/Attorney_General_of_Ireland",
        "excerpt": "In Ireland, the Chief State Solicitor's Office provides litigation and solicitor services on behalf of the Attorney General and Government Departments.",
    },
    "localnewsqa_ambig_0034971": {
        "title": "Minister for Foreign Affairs (Ireland)",
        "url": "https://en.wikipedia.org/wiki/Minister_for_Foreign_Affairs_(Ireland)",
        "excerpt": "In Ireland, the Minister for Foreign Affairs is a senior minister. Eamon Gilmore served as Tánaiste and Minister for Foreign Affairs and Trade from 2011 to 2014.",
    },
    "localnewsqa_explicit_0003048": {
        "title": "Rebel Salute",
        "url": "https://en.wikipedia.org/wiki/Rebel_Salute",
        "excerpt": "Rebel Salute is an annual reggae festival in Jamaica. It has been associated with St Ann's Bay and the parish of Saint Ann in Jamaican entertainment coverage.",
    },
    "localnewsqa_ambig_0021564": {
        "title": "List of Jamaican high school football champions",
        "url": "https://en.wikipedia.org/wiki/List_of_Jamaican_high_school_football_champions",
        "excerpt": "The daCosta Cup is a Jamaican schoolboy football competition. It is contested by rural-area high schools in Jamaica.",
    },
    "localnewsqa_explicit_0014863": {
        "title": "Nairobi Expressway",
        "url": "https://en.wikipedia.org/wiki/Nairobi_Expressway",
        "excerpt": "The Nairobi Expressway is a toll road in Kenya. The route runs along the Mombasa Road corridor through Nairobi.",
    },
    "localnewsqa_explicit_0009340": {
        "title": "Cabinet of Malaysia",
        "url": "https://en.wikipedia.org/wiki/Cabinet_of_Malaysia",
        "excerpt": "In Malaysia, the Ministry of Agriculture and Agro-Based Industry was headed by Ahmad Shabery Cheek in the Najib Razak cabinet.",
    },
    "localnewsqa_explicit_0010175": {
        "title": "Road Transport Department Malaysia",
        "url": "https://en.wikipedia.org/wiki/Road_Transport_Department_Malaysia",
        "excerpt": "The Road Transport Department (JPJ) is Malaysia's road transport department, responsible for road transport enforcement and driver and vehicle licensing.",
    },
    "localnewsqa_ambig_0027672": {
        "title": "Public holidays in Malaysia",
        "url": "https://en.wikipedia.org/wiki/Public_holidays_in_Malaysia",
        "excerpt": "Public holidays in Malaysia include Hari Keputeraan Yang di-Pertuan Agong, the official birthday celebration of the Yang di-Pertuan Agong.",
    },
    "localnewsqa_explicit_0012328": {
        "title": "Rangers International F.C.",
        "url": "https://en.wikipedia.org/wiki/Rangers_International_F.C.",
        "excerpt": "Rangers International F.C. is a Nigerian football club based in Enugu. Its home ground is Nnamdi Azikiwe Stadium, Enugu.",
    },
    "localnewsqa_explicit_0011240": {
        "title": "Public holidays in the Philippines",
        "url": "https://en.wikipedia.org/wiki/Public_holidays_in_the_Philippines",
        "excerpt": "Public holidays in the Philippines include Eid'l Adha, the Feast of Sacrifice, which is declared by presidential proclamation for the Muslim holiday.",
    },
    "localnewsqa_explicit_0007386": {
        "title": "GCE Advanced Level in Sri Lanka",
        "url": "https://en.wikipedia.org/wiki/GCE_Advanced_Level_in_Sri_Lanka",
        "excerpt": "The Sri Lankan Advanced Level, commonly called the G.C.E. Advanced Level, is an examination in Sri Lanka used for university entrance and undergraduate selection.",
    },
    "localnewsqa_explicit_0016557": {
        "title": "A-level",
        "url": "https://en.wikipedia.org/wiki/A-level",
        "excerpt": "The Advanced Certificate of Secondary Education Examination (ACSEE) is an advanced-level secondary education examination in Tanzania taken after Form Six.",
    },
    "localnewsqa_explicit_0016039": {
        "title": "Chama Cha Mapinduzi",
        "url": "https://en.wikipedia.org/wiki/Chama_Cha_Mapinduzi",
        "excerpt": "Chama Cha Mapinduzi (CCM) is the dominant ruling party in Tanzania and has held the presidency and a parliamentary majority through the multiparty era.",
    },
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
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def write_fetch_cache(path: Path, fetches: dict[str, dict], selected_urls: set[str]) -> None:
    rows = [fetches[url] for url in sorted(selected_urls) if url in fetches]
    write_jsonl(path, rows)


def norm(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    explicit_audit = load_module(AUDIT_EXPLICIT_SCRIPT, "audit48")
    fetch_audit = load_module(FETCH_AUDIT_SCRIPT, "fetch32")
    builder = load_module(BUILDER_SCRIPT, "builder45")
    rows = read_jsonl(INPUT_JSONL)
    ambiguous_ids = {row["source_row_id"] for row in read_jsonl(AMBIGUOUS)}
    fetches: dict[str, dict] = {}
    for cache in SEED_FETCH_CACHES:
        fetches.update(explicit_audit.load_fetch_cache(cache))

    repair_log = []
    for row in rows:
        source_id = row.get("source_row_id", "")
        if source_id not in REPAIRS:
            continue
        repair = REPAIRS[source_id]
        old = {
            "source_row_id": source_id,
            "country": row.get("country", ""),
            "question": row.get("question", ""),
            "target_answer": row.get("target_answer", ""),
            "old_title": row.get("target_evidence_title", ""),
            "old_url": row.get("target_evidence_url", ""),
            "old_excerpt": row.get("target_evidence_excerpt", ""),
            "new_title": repair["title"],
            "new_url": repair["url"],
            "new_excerpt": repair["excerpt"],
        }
        row["target_evidence_title"] = repair["title"]
        row["target_evidence_url"] = repair["url"]
        row["target_evidence_excerpt"] = repair["excerpt"]
        if repair["url"] not in fetches or not fetches.get(repair["url"], {}).get("ok"):
            fetched = fetch_audit.fetch_url(repair["url"], timeout=25, delay=0)
            fetches[repair["url"]] = fetched
        repair_log.append(old)

    failures = []
    warnings = []
    high_conf_rows = []
    anchor_rows = []
    for row in rows:
        row_failures, row_warnings = explicit_audit.audit_explicit_row(row, fetches, ambiguous_ids, fetch_audit)
        if row_failures:
            failures.append(
                {
                    "source_row_id": row.get("source_row_id", ""),
                    "country": row.get("country", ""),
                    "failures": " | ".join(row_failures),
                    "warnings": " | ".join(row_warnings),
                    "target_evidence_title": row.get("target_evidence_title", ""),
                    "target_evidence_url": row.get("target_evidence_url", ""),
                }
            )
        if row_warnings:
            warnings.append(
                {
                    "source_row_id": row.get("source_row_id", ""),
                    "country": row.get("country", ""),
                    "warnings": " | ".join(row_warnings),
                    "target_evidence_title": row.get("target_evidence_title", ""),
                    "target_evidence_url": row.get("target_evidence_url", ""),
                }
            )
        cue_text = builder.evidence_cue_text(row, "target", fetches)
        high_conf_issues = [cue for cue in builder.high_confidence_question_cues(row) if cue["norm"] not in cue_text]
        if high_conf_issues:
            high_conf_rows.append(
                {
                    "source_row_id": row.get("source_row_id", ""),
                    "country": row.get("country", ""),
                    "issues": " | ".join(cue["cue"] for cue in high_conf_issues),
                }
            )
        anchor_text = " ".join(
            str(part or "")
            for part in [
                row.get("target_evidence_title", ""),
                row.get("target_evidence_excerpt", ""),
                fetches.get(row.get("target_evidence_url", ""), {}).get("title", ""),
                str(fetches.get(row.get("target_evidence_url", ""), {}).get("text", ""))[:1500],
            ]
        )
        anchor_norm = fetch_audit.normalize_text(anchor_text)
        if not fetch_audit.contains_any(anchor_norm, fetch_audit.answer_aliases(row.get("target_answer", ""))):
            anchor_rows.append(
                {
                    "source_row_id": row.get("source_row_id", ""),
                    "country": row.get("country", ""),
                    "target_answer": row.get("target_answer", ""),
                    "target_evidence_title": row.get("target_evidence_title", ""),
                }
            )

    source_ids = [row["source_row_id"] for row in rows]
    question_keys = [norm(row["question"]) for row in rows]
    selected_urls = {row.get("target_evidence_url", "") for row in rows if row.get("target_evidence_url", "")}
    write_jsonl(INPUT_JSONL, rows)
    write_csv(INPUT_CSV, rows)
    write_fetch_cache(FETCH_CACHE, fetches, selected_urls)
    write_csv(REPAIR_LOG, repair_log)
    write_csv(STRICT_DIR / "answer_anchor_manual_evidence_repair_failures.csv", failures)
    write_csv(STRICT_DIR / "answer_anchor_manual_evidence_repair_warnings.csv", warnings)
    write_csv(STRICT_DIR / "answer_anchor_manual_evidence_repair_high_conf_rows.csv", high_conf_rows)
    write_csv(STRICT_DIR / "answer_anchor_manual_evidence_repair_anchor_rows.csv", anchor_rows)

    summary = {
        "rows": len(rows),
        "repairs": len(repair_log),
        "failure_rows": len(failures),
        "warning_rows": len(warnings),
        "high_confidence_question_cue_rows": len(high_conf_rows),
        "answer_anchor_rows": len(anchor_rows),
        "country_counts": dict(Counter(row["country"] for row in rows)),
        "duplicate_source_ids": len(source_ids) - len(set(source_ids)),
        "duplicate_questions": len(question_keys) - len(set(question_keys)),
        "ambiguous_overlap": len(set(source_ids) & ambiguous_ids),
        "valid": not failures and not warnings and not high_conf_rows and not anchor_rows,
        "paths": {
            "jsonl": str(INPUT_JSONL),
            "csv": str(INPUT_CSV),
            "fetch_cache": str(FETCH_CACHE),
            "repair_log": str(REPAIR_LOG),
            "summary": str(SUMMARY),
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
