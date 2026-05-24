#!/usr/bin/env python3

import argparse
import csv
import html
import importlib.util
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import requests


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515"
DEFAULT_POOL = BASE / "localnewsqa_ambiguous_verifiable_pool_4625.jsonl"
DEFAULT_FETCH_CACHE = BASE / "web_audit_ambiguous_1700/url_fetches.jsonl"
DEFAULT_OUTDIR = BASE / "web_semantic_gold_ambiguous_1700"
AUDIT_SCRIPT = ROOT / "qa_data/localnewsqa_core/32_web_audit_ambiguous_verifiable.py"

COUNTRY_ORDER = [
    "United States",
    "Canada",
    "Jamaica",
    "India",
    "Pakistan",
    "Bangladesh",
    "Sri Lanka",
    "Hong Kong",
    "Malaysia",
    "Philippines",
    "Nigeria",
    "South Africa",
    "Kenya",
    "Ghana",
    "Tanzania",
    "United Kingdom",
    "Ireland",
]

GENERIC_SLUGS = {
    "president",
    "prime_minister",
    "parliament",
    "senate",
    "national_assembly",
    "house_of_representatives",
    "governor",
    "chair",
    "commissioner",
    "director",
    "mayor",
    "cabinet",
    "monarch",
    "high_court",
    "supreme_court",
    "attorney_general",
    "attorney-general",
    "chief_justice",
    "city_corporation",
    "municipal_corporation",
    "municipal_council",
    "district_court",
    "court_of_appeal",
    "speaker",
    "governor-general",
    "currency",
    "dollar",
    "rupee",
    "rand",
    "euro",
    "euros",
    "cent",
    "cents",
    "paisa",
    "taka",
    "primary_school",
    "secondary_school",
    "high_school",
    "college",
    "university",
    "school",
    "schools",
    "primary",
    "secondary",
    "elementary_school",
    "public_schools",
    "state_schools",
    "association_football",
    "football",
    "cricket",
    "rugby",
    "rugby_league",
    "basketball",
    "marathon",
    "race",
    "left",
    "right",
    "independence",
    "thanksgiving",
    "culture",
    "blues",
    "rock",
    "dancehall",
    "teacher",
    "teaching",
    "trades",
    "course",
    "vocational_school",
    "public_holiday",
    "bank_holiday",
    "holiday",
}

OVERRIDES = {
    ("United States", "President"): "President of the United States",
    ("United States", "Vice President"): "Vice President of the United States",
    ("United States", "Congress"): "United States Congress",
    ("United States", "House of Representatives"): "United States House of Representatives",
    ("United States", "Senate"): "United States Senate",
    ("United States", "District Court"): "United States district court",
    ("United States", "Chief Justice"): "Chief Justice of the United States",
    ("United States", "Solicitor General"): "Solicitor General of the United States",
    ("United States", "Austin"): "Austin, Texas",
    ("United States", "Thanksgiving"): "Thanksgiving (United States)",
    ("United States", "The Capitol"): "United States Capitol",
    ("United States", "ABC"): "American Broadcasting Company",
    ("United States", "AP"): "Advanced Placement",
    ("United States", "Democratic Party"): "Democratic Party (United States)",
    ("United Kingdom", "Prime Minister"): "Prime Minister of the United Kingdom",
    ("United Kingdom", "Supreme Court"): "Supreme Court of the United Kingdom",
    ("United Kingdom", "Attorney General"): "Attorney General for England and Wales",
    ("United Kingdom", "National Audit Office"): "National Audit Office (United Kingdom)",
    ("United Kingdom", "Cabinet"): "Cabinet of the United Kingdom",
    ("United Kingdom", "King's Cross station"): "London King's Cross railway station",
    ("United Kingdom", "Armistice remembrance"): "Armistice Day",
    ("United Kingdom", "May Day bank holiday"): "May Day",
    ("United Kingdom", "Spring Bank Holiday"): "Spring Bank Holiday",
    ("United Kingdom", "Serious Fraud Office"): "Serious Fraud Office (United Kingdom)",
    ("United Kingdom", "Boundary Commission"): "Boundary Commissions (United Kingdom)",
    ("United Kingdom", "Red Roses"): "England women's national rugby union team",
    ("Jamaica", "Kingston"): "Kingston, Jamaica",
    ("Jamaica", "Parliament"): "Parliament of Jamaica",
    ("Jamaica", "Supreme Court"): "Supreme Court of Judicature of Jamaica",
    ("Jamaica", "Governor-General"): "Governor-General of Jamaica",
    ("Jamaica", "Attorney General"): "Attorney-General of Jamaica",
    ("Jamaica", "Prime Minister"): "Prime Minister of Jamaica",
    ("Jamaica", "Gordon House"): "Gordon House (Jamaica)",
    ("Jamaica", "King's House"): "King's House, Jamaica",
    ("Jamaica", "TVJ"): "Television Jamaica",
    ("Jamaica", "Lucea"): "Lucea, Jamaica",
    ("Jamaica", "Morant Bay"): "Morant Bay",
    ("Hong Kong", "Legislative Council"): "Legislative Council of Hong Kong",
    ("Hong Kong", "Equal Opportunities Commission"): "Equal Opportunities Commission (Hong Kong)",
    ("Hong Kong", "Audit Commission"): "Audit Commission (Hong Kong)",
    ("Hong Kong", "Competition Commission"): "Competition Commission (Hong Kong)",
    ("Hong Kong", "Security Bureau"): "Security Bureau (Hong Kong)",
    ("Hong Kong", "Chief Secretary"): "Chief Secretary for Administration",
    ("Hong Kong", "ICAC"): "Independent Commission Against Corruption (Hong Kong)",
    ("Hong Kong", "Independent Commission Against Corruption"): "Independent Commission Against Corruption (Hong Kong)",
    ("Hong Kong", "Inland Revenue Department"): "Inland Revenue Department (Hong Kong)",
    ("Hong Kong", "Policy Address"): "Policy Address (Hong Kong)",
    ("Hong Kong", "DSE"): "Hong Kong Diploma of Secondary Education",
    ("Hong Kong", "TVB"): "Television Broadcasts Limited",
    ("Hong Kong", "ATV"): "Asia Television",
    ("Hong Kong", "Tiananmen vigil"): "Hong Kong Alliance in Support of Patriotic Democratic Movements of China",
    ("Kenya", "National Assembly"): "National Assembly (Kenya)",
    ("Kenya", "Senate"): "Senate of Kenya",
    ("Kenya", "Attorney General"): "Attorney-General of Kenya",
    ("Kenya", "Attorney-General"): "Attorney-General of Kenya",
    ("Kenya", "Judicial Service Commission"): "Judicial Service Commission (Kenya)",
    ("Kenya", "KCSE"): "Kenya Certificate of Secondary Education",
    ("Kenya", "KNUT"): "Kenya National Union of Teachers",
    ("Kenya", "The Standard"): "The Standard (Kenya)",
    ("Kenya", "State House"): "State House, Nairobi",
    ("Kenya", "Tana"): "Tana River (Kenya)",
    ("Kenya", "Jubilee Party"): "Jubilee Party of Kenya",
    ("Kenya", "Equity Bank"): "Equity Group Holdings",
    ("Kenya", "Simbas"): "Kenya national rugby union team",
    ("Kenya", "Bandari"): "Bandari F.C.",
    ("Kenya", "Public Service Commission"): "Public Service Commission (Kenya)",
    ("Malaysia", "Federal Court"): "Federal Court of Malaysia",
    ("Malaysia", "Astro"): "Astro (television)",
    ("Malaysia", "RTM"): "Radio Televisyen Malaysia",
    ("Malaysia", "KUL"): "Kuala Lumpur International Airport",
    ("Malaysia", "LPF"): "Film Censorship Board of Malaysia",
    ("Malaysia", "UMNO"): "United Malays National Organisation",
    ("Malaysia", "Parliament"): "Parliament of Malaysia",
    ("Malaysia", "Petronas"): "Petronas",
    ("Malaysia", "Cameron Highlands"): "Cameron Highlands",
    ("Malaysia", "Johor Bahru"): "Johor Bahru",
    ("Nigeria", "National Assembly"): "National Assembly (Nigeria)",
    ("Nigeria", "Senate"): "Senate of Nigeria",
    ("Nigeria", "House of Representatives"): "House of Representatives (Nigeria)",
    ("Nigeria", "President"): "President of Nigeria",
    ("Nigeria", "APC"): "All Progressives Congress",
    ("Nigeria", "PDP"): "Peoples Democratic Party (Nigeria)",
    ("India", "President"): "President of India",
    ("India", "Rupee"): "Indian rupee",
    ("India", "INR"): "Indian rupee",
    ("Bangladesh", "Taka"): "Bangladeshi taka",
    ("Bangladesh", "BDT"): "Bangladeshi taka",
    ("Bangladesh", "DSE"): "Dhaka Stock Exchange",
    ("Bangladesh", "BSE"): "Bombay Stock Exchange",
    ("Bangladesh", "Tofail Ahmed"): "Tofail Ahmed (politician)",
    ("Ireland", "Michael Noonan"): "Michael Noonan (Fine Gael politician)",
    ("Philippines", "Liberal Party"): "Liberal Party (Philippines)",
}

URL_OVERRIDES = {
    ("Jamaica", "Auditor General’s Department"): "https://auditorgeneral.gov.jm/",
    ("Jamaica", "Auditor General's Department"): "https://auditorgeneral.gov.jm/",
    ("Jamaica", "Supreme Court"): "https://supremecourt.gov.jm/",
    ("Jamaica", "Attorney General"): "https://www.agc.gov.jm/",
    ("Jamaica", "Office of the Services Commissions"): "https://osc.gov.jm/",
    ("Jamaica", "Ministry of Health and Wellness"): "https://www.moh.gov.jm/",
    ("Jamaica", "Fair Trading Commission"): "https://jftc.gov.jm/",
    ("Jamaica", "PEP"): "https://moey.gov.jm/primary-exit-profile-pep/",
    ("Jamaica", "CAPE"): "https://www.cxc.org/examinations/cape/",
}


def load_audit_module():
    spec = importlib.util.spec_from_file_location("audit32", AUDIT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def normalize_text(text: Any) -> str:
    text = html.unescape(str(text or "")).lower()
    text = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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


def title_to_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe="()_,:'-&.")


def compact_text(text: str, limit: int = 120_000) -> str:
    text = re.sub(r"\s+", " ", html.unescape(str(text or ""))).strip()
    return text[:limit]


def fetch_titles(titles: set[str], sleep: float = 1.0) -> dict[str, dict]:
    titles = sorted(titles)
    out = {}
    session = requests.Session()
    session.headers.update({"User-Agent": "LocalNewsQA-semantic-gold-title-fetch/1.0"})
    batch_size = 20
    for start in range(0, len(titles), batch_size):
        batch = titles[start : start + batch_size]
        response = None
        error = ""
        for attempt in range(6):
            try:
                response = session.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "format": "json",
                        "formatversion": "2",
                        "redirects": "1",
                        "prop": "extracts|info|pageprops",
                        "inprop": "url",
                        "explaintext": "1",
                        "exintro": "1",
                        "exlimit": "max",
                        "titles": "|".join(batch),
                    },
                    timeout=30,
                )
                if response.status_code == 429:
                    raise requests.HTTPError("429 rate limited", response=response)
                response.raise_for_status()
                error = ""
                break
            except Exception as exc:
                error = str(exc)[:500]
                if attempt < 5:
                    time.sleep(3.0 * (attempt + 1))
        if error:
            for title in batch:
                out[title] = {"ok": False, "title": title, "url": title_to_url(title), "error": error}
            continue
        data = response.json()
        pages = data.get("query", {}).get("pages", [])
        page_by_title = {page.get("title", ""): page for page in pages}
        for item in data.get("query", {}).get("normalized", []):
            if item.get("from") in batch and item.get("to") in page_by_title:
                page_by_title[item["from"]] = page_by_title[item["to"]]
        for item in data.get("query", {}).get("redirects", []):
            if item.get("from") in batch and item.get("to") in page_by_title:
                page_by_title[item["from"]] = page_by_title[item["to"]]
        for title in batch:
            page = page_by_title.get(title)
            if not page or page.get("missing"):
                out[title] = {"ok": False, "title": title, "url": title_to_url(title), "error": "missing"}
                continue
            text = compact_text(page.get("extract", ""))
            if "disambiguation" in page.get("pageprops", {}):
                text = f"{text} This disambiguation page lists articles associated with the title {page.get('title', title)}."
            out[title] = {
                "ok": bool(text),
                "title": page.get("title", title),
                "url": page.get("fullurl") or title_to_url(page.get("title", title)),
                "text": text,
                "error": "" if text else "empty",
            }
        print(f"override title batch {start + len(batch)}/{len(titles)}", flush=True)
        time.sleep(sleep)
    return out


def fetch_row_for_page(url: str, page: dict) -> dict:
    return {
        "url": url,
        "ok": bool(page.get("ok")),
        "status_code": 200 if page.get("ok") else "",
        "final_url": page.get("url", url),
        "content_type": "application/json; mediawiki extract",
        "title": page.get("title", ""),
        "text": page.get("text", ""),
        "text_len": len(page.get("text", "")),
        "error": page.get("error", ""),
        "elapsed_sec": 0.0,
        "source": "semantic_gold_override_extract",
    }


def fetch_direct_url(url: str) -> dict:
    started = time.time()
    row = {
        "url": url,
        "ok": False,
        "status_code": "",
        "final_url": url,
        "content_type": "",
        "title": "",
        "text": "",
        "text_len": 0,
        "error": "",
        "elapsed_sec": 0.0,
        "source": "semantic_gold_url_override_direct",
    }
    try:
        response = requests.get(url, timeout=25, headers={"User-Agent": "LocalNewsQA-semantic-gold-url-fetch/1.0"})
        row["status_code"] = response.status_code
        row["final_url"] = response.url
        row["content_type"] = response.headers.get("content-type", "")
        response.raise_for_status()
        raw = response.text
        title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, flags=re.IGNORECASE | re.DOTALL)
        title = re.sub(r"<[^>]+>", " ", title_match.group(1)).strip() if title_match else ""
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        row["title"] = html.unescape(title)
        row["text"] = compact_text(text)
        row["text_len"] = len(row["text"])
        row["ok"] = bool(row["text"])
    except Exception as exc:
        row["error"] = str(exc)[:500]
    row["elapsed_sec"] = round(time.time() - started, 3)
    return row


def wiki_title_from_url(url: str) -> str:
    if "wikipedia.org/wiki/" not in url:
        return ""
    raw = url.split("/wiki/", 1)[1].split("#", 1)[0]
    return unquote(raw).replace("_", " ")


def fetch_full_wikipedia_pages(urls: set[str], sleep: float = 0.2) -> dict[str, dict]:
    out = {}
    session = requests.Session()
    session.headers.update({"User-Agent": "LocalNewsQA-semantic-gold-full-refresh/1.0"})
    for url in sorted(urls):
        title = wiki_title_from_url(url)
        row = {
            "url": url,
            "ok": False,
            "status_code": "",
            "final_url": url,
            "content_type": "application/json; mediawiki full extract",
            "title": title,
            "text": "",
            "text_len": 0,
            "error": "",
            "elapsed_sec": 0.0,
            "source": "semantic_gold_full_extract_refresh",
        }
        started = time.time()
        try:
            response = session.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "format": "json",
                    "formatversion": "2",
                    "redirects": "1",
                    "prop": "extracts|info|pageprops",
                    "inprop": "url",
                    "explaintext": "1",
                    "titles": title,
                },
                timeout=30,
            )
            row["status_code"] = response.status_code
            response.raise_for_status()
            data = response.json()
            pages = data.get("query", {}).get("pages", [])
            page = pages[0] if pages else {}
            if page and not page.get("missing"):
                text = compact_text(page.get("extract", ""))
                if "disambiguation" in page.get("pageprops", {}):
                    text = f"{text} This disambiguation page lists articles associated with the title {page.get('title', title)}."
                row.update(
                    {
                        "ok": bool(text),
                        "final_url": page.get("fullurl") or url,
                        "title": page.get("title", title),
                        "text": text,
                        "text_len": len(text),
                        "error": "" if text else "empty",
                    }
                )
            else:
                row["error"] = "missing"
        except Exception as exc:
            row["error"] = str(exc)[:500]
        row["elapsed_sec"] = round(time.time() - started, 3)
        out[url] = row
        time.sleep(sleep)
    return out


def slug(url: str) -> str:
    raw = url.rstrip("/").split("/wiki/")[-1] if "/wiki/" in url else url.rstrip("/").split("/")[-1]
    return raw.lower().replace("%27", "'")


def is_disambiguation(fetch: dict) -> bool:
    text = (fetch.get("text") or "").lower()
    return (
        "this disambiguation page lists articles associated with the title" in text
        or "this disambiguation page lists articles associated with the same title" in text
    )


def side_risk(row: dict, side: str, fetches: dict[str, dict]) -> list[str]:
    url = row.get(f"{side}_evidence_url", "")
    out = []
    if is_disambiguation(fetches.get(url, {})):
        out.append("disambiguation_evidence")
    if slug(url) in GENERIC_SLUGS:
        out.append("generic_evidence")
    return out


def apply_overrides(row: dict, pages: dict[str, dict], url_pages: dict[str, dict], fetches: dict[str, dict]) -> tuple[dict, list[dict]]:
    out = dict(row)
    logs = []
    for side in ["target", "contrast"]:
        country = out["country"] if side == "target" else out.get("contrast_country", "")
        answer = out.get(f"{side}_answer", "")
        title = OVERRIDES.get((country, answer))
        direct_url = URL_OVERRIDES.get((country, answer))
        if not title:
            if not direct_url:
                continue
            direct = url_pages.get(direct_url, {})
            if not direct.get("ok"):
                logs.append({"source_row_id": out["source_row_id"], "side": side, "status": "url_override_missing", "title": answer, "new_url": direct_url})
                continue
            old_url = out.get(f"{side}_evidence_url", "")
            out[f"{side}_evidence_url"] = direct_url
            out[f"{side}_evidence_title"] = direct.get("title") or answer
            out[f"{side}_evidence_excerpt"] = direct.get("text", "")[:1000]
            fetches[direct_url] = direct
            logs.append(
                {
                    "source_row_id": out["source_row_id"],
                    "side": side,
                    "status": "url_override_applied",
                    "old_url": old_url,
                    "new_url": direct_url,
                    "title": direct.get("title") or answer,
                }
            )
        else:
            page = pages.get(title, {})
            if not page.get("ok"):
                if direct_url:
                    direct = url_pages.get(direct_url, {})
                    if not direct.get("ok"):
                        logs.append({"source_row_id": out["source_row_id"], "side": side, "status": "url_override_missing", "title": answer, "new_url": direct_url})
                        continue
                    old_url = out.get(f"{side}_evidence_url", "")
                    out[f"{side}_evidence_url"] = direct_url
                    out[f"{side}_evidence_title"] = direct.get("title") or answer
                    out[f"{side}_evidence_excerpt"] = direct.get("text", "")[:1000]
                    fetches[direct_url] = direct
                    logs.append(
                        {
                            "source_row_id": out["source_row_id"],
                            "side": side,
                            "status": "url_override_applied_after_title_missing",
                            "old_url": old_url,
                            "new_url": direct_url,
                            "title": direct.get("title") or answer,
                            "missing_title": title,
                        }
                    )
                    continue
                logs.append({"source_row_id": out["source_row_id"], "side": side, "status": "override_missing", "title": title})
                continue
            url = page["url"]
            old_url = out.get(f"{side}_evidence_url", "")
            out[f"{side}_evidence_url"] = url
            out[f"{side}_evidence_title"] = page["title"]
            out[f"{side}_evidence_excerpt"] = page["text"][:1000]
            fetches[url] = fetch_row_for_page(url, page)
            logs.append(
                {
                    "source_row_id": out["source_row_id"],
                    "side": side,
                    "status": "override_applied",
                    "old_url": old_url,
                    "new_url": url,
                    "title": page["title"],
                }
            )
    return out, logs


def acceptable_audit(audit: dict) -> bool:
    if audit["severity"] == "pass":
        return True
    if audit["severity"] != "warn":
        return False
    warnings = {part.strip() for part in audit["warnings"].split(" | ") if part.strip()}
    return bool(warnings) and warnings <= {"target_evidence_text_short", "contrast_evidence_text_short"}


def selection_key(row: dict) -> tuple:
    return (
        int(row.get("semantic_gold_warning_count", 0) > 0),
        0 if row.get("weak_locale_risk") == "clean" else 1,
        0 if str(row.get("llm_accept_or_restored", "")).lower() == "true" else 1,
        -float(row.get("evidence_support_score", 0) or 0),
        float(row.get("semantic_error_probability", 1) or 1),
        row.get("source_row_id", ""),
    )


def validate(rows: list[dict]) -> list[str]:
    errors = []
    counts = Counter(row["country"] for row in rows)
    if len(rows) != 1700:
        errors.append(f"expected 1700 rows, got {len(rows)}")
    for country in COUNTRY_ORDER:
        if counts[country] != 100:
            errors.append(f"{country}: expected 100, got {counts[country]}")
    source_ids = [row["source_row_id"] for row in rows]
    if len(source_ids) != len(set(source_ids)):
        errors.append("duplicate source ids")
    return errors


def refresh_selected_short_wikipedia_evidence(final_rows: list[dict], fetches: dict[str, dict], audit_mod: Any) -> list[dict]:
    short_urls = set()
    refresh_logs = []
    for row in final_rows:
        audit = audit_mod.audit_row(row, fetches)
        warnings = {part.strip() for part in audit["warnings"].split(" | ") if part.strip()}
        for side in ["target", "contrast"]:
            if f"{side}_evidence_text_short" in warnings:
                url = row.get(f"{side}_evidence_url", "")
                if wiki_title_from_url(url):
                    short_urls.add(url)

    refreshed = fetch_full_wikipedia_pages(short_urls) if short_urls else {}
    for url, row in refreshed.items():
        old = fetches.get(url, {})
        if row.get("ok") and row.get("text_len", 0) >= old.get("text_len", 0):
            fetches[url] = row
            status = "refreshed"
        else:
            status = "kept_existing"
        refresh_logs.append(
            {
                "url": url,
                "status": status,
                "old_text_len": old.get("text_len", 0),
                "new_text_len": row.get("text_len", 0),
                "error": row.get("error", ""),
            }
        )

    for row in final_rows:
        audit = audit_mod.audit_row(row, fetches)
        warning_parts = [part.strip() for part in audit["warnings"].split(" | ") if part.strip()]
        row["semantic_gold_notes"] = audit["warnings"]
        row["semantic_gold_mechanical_web_audit_severity"] = audit["severity"]
        row["semantic_gold_warning_count"] = len(warning_parts)
        row["web_audit_severity"] = "pass" if acceptable_audit(audit) else audit["severity"]
        row["web_audit_failures"] = "" if acceptable_audit(audit) else audit["failures"]
        row["web_audit_warnings"] = audit["warnings"]
    return refresh_logs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build semantic-gold ambiguous split from pool.")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--fetch-cache", type=Path, default=DEFAULT_FETCH_CACHE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    audit_mod = load_audit_module()
    rows = read_jsonl(args.pool)
    fetches = audit_mod.load_existing_fetches(args.fetch_cache)
    titles = set(OVERRIDES.values())
    pages = fetch_titles(titles)
    url_pages = {url: fetch_direct_url(url) for url in sorted(set(URL_OVERRIDES.values()))}

    candidates_by_country: dict[str, list[dict]] = defaultdict(list)
    rejected = []
    override_logs = []
    for row in rows:
        repaired, logs = apply_overrides(row, pages, url_pages, fetches)
        override_logs.extend(logs)
        audit = audit_mod.audit_row(repaired, fetches)
        risks = side_risk(repaired, "target", fetches) + side_risk(repaired, "contrast", fetches)
        if acceptable_audit(audit) and not risks:
            out = dict(repaired)
            warning_parts = [part.strip() for part in audit["warnings"].split(" | ") if part.strip()]
            out["semantic_gold_status"] = "pass"
            out["semantic_gold_notes"] = audit["warnings"]
            out["semantic_gold_mechanical_web_audit_severity"] = audit["severity"]
            out["semantic_gold_warning_count"] = len(warning_parts)
            out["web_audit_severity"] = "pass"
            out["web_audit_failures"] = ""
            out["web_audit_warnings"] = audit["warnings"]
            candidates_by_country[out["country"]].append(out)
        else:
            rejected.append(
                {
                    "source_row_id": row["source_row_id"],
                    "country": row["country"],
                    "audit_severity": audit["severity"],
                    "audit_failures": audit["failures"],
                    "audit_warnings": audit["warnings"],
                    "semantic_risks": " | ".join(risks),
                    "question": row["question"],
                    "target_answer": row["target_answer"],
                    "contrast_answer": row["contrast_answer"],
                    "target_evidence_url": repaired["target_evidence_url"],
                    "contrast_evidence_url": repaired["contrast_evidence_url"],
                }
            )

    selected = []
    deficits = {}
    for country in COUNTRY_ORDER:
        pool = sorted(candidates_by_country.get(country, []), key=selection_key)
        if len(pool) < 100:
            deficits[country] = 100 - len(pool)
        selected.extend(pool[:100])

    selected.sort(key=lambda row: (COUNTRY_ORDER.index(row["country"]), selection_key(row)))
    final_rows = []
    for idx, row in enumerate(selected, start=1):
        out = dict(row)
        out["id"] = f"localnewsqa_ambig_semantic_gold_{idx:04d}"
        out["split_name"] = "LocalNewsQA-Ambiguous-Semantic-Gold-1700"
        final_rows.append(out)

    refresh_logs = refresh_selected_short_wikipedia_evidence(final_rows, fetches, audit_mod)
    errors = validate(final_rows)
    post_refresh_bad = []
    for row in final_rows:
        audit = audit_mod.audit_row(row, fetches)
        risks = side_risk(row, "target", fetches) + side_risk(row, "contrast", fetches)
        if not acceptable_audit(audit) or risks:
            post_refresh_bad.append(
                {
                    "source_row_id": row["source_row_id"],
                    "country": row["country"],
                    "audit_severity": audit["severity"],
                    "audit_failures": audit["failures"],
                    "audit_warnings": audit["warnings"],
                    "semantic_risks": " | ".join(risks),
                }
            )
    if post_refresh_bad:
        errors.append(f"post-refresh invalid selected rows: {len(post_refresh_bad)}")
    if deficits:
        errors.append(f"semantic candidate deficits: {deficits}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.outdir / "localnewsqa_ambiguous_semantic_gold_1700.jsonl"
    csv_path = args.outdir / "localnewsqa_ambiguous_semantic_gold_1700.csv"
    rejected_path = args.outdir / "semantic_gold_rejected_pool_rows.csv"
    override_path = args.outdir / "semantic_gold_override_log.csv"
    refresh_path = args.outdir / "semantic_gold_full_extract_refresh_log.csv"
    post_refresh_bad_path = args.outdir / "semantic_gold_post_refresh_invalid_rows.csv"
    selected_fetch_path = args.outdir / "semantic_gold_selected_evidence_fetches.jsonl"
    summary_path = args.outdir / "summary.json"
    write_jsonl(jsonl_path, final_rows)
    write_csv(csv_path, final_rows)
    write_csv(rejected_path, rejected)
    write_csv(override_path, override_logs)
    write_csv(refresh_path, refresh_logs)
    write_csv(post_refresh_bad_path, post_refresh_bad)
    selected_urls = sorted(
        {
            url
            for row in final_rows
            for url in [row.get("target_evidence_url", ""), row.get("contrast_evidence_url", "")]
            if url
        }
    )
    write_jsonl(selected_fetch_path, [fetches[url] for url in selected_urls if url in fetches])
    summary = {
        "pool": str(args.pool),
        "rows": len(final_rows),
        "valid": not errors,
        "validation_errors": errors,
        "candidate_counts_by_country": {country: len(candidates_by_country.get(country, [])) for country in COUNTRY_ORDER},
        "country_counts": dict(Counter(row["country"] for row in final_rows)),
        "override_counts": dict(Counter(row["status"] for row in override_logs)),
        "candidate_mechanical_web_audit_severity_counts": dict(
            Counter(row.get("semantic_gold_mechanical_web_audit_severity", "") for rows in candidates_by_country.values() for row in rows)
        ),
        "selected_mechanical_web_audit_severity_counts": dict(
            Counter(row.get("semantic_gold_mechanical_web_audit_severity", "") for row in final_rows)
        ),
        "selected_web_audit_warning_counts": dict(Counter(row.get("web_audit_warnings", "") for row in final_rows)),
        "selected_warning_rows": sum(1 for row in final_rows if row.get("web_audit_warnings")),
        "full_extract_refresh_counts": dict(Counter(row.get("status", "") for row in refresh_logs)),
        "selected_with_overrides": sum(
            1
            for row in final_rows
            for title in OVERRIDES.values()
            if row.get("target_evidence_title") == title or row.get("contrast_evidence_title") == title
        ),
        "paths": {
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
            "rejected": str(rejected_path),
            "override_log": str(override_path),
            "full_extract_refresh_log": str(refresh_path),
            "post_refresh_invalid_rows": str(post_refresh_bad_path),
            "selected_evidence_fetches": str(selected_fetch_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
