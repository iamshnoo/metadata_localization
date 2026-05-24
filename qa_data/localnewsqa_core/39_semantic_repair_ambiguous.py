#!/usr/bin/env python3

import argparse
import csv
import html
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


ROOT = Path(__file__).resolve().parents[2]
STRICT_DIR = ROOT / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_strict_pass_ambiguous_1700"
DEFAULT_INPUT = STRICT_DIR / "localnewsqa_ambiguous_web_repaired_1700.jsonl"
DEFAULT_FETCH_CACHE = (
    ROOT / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_audit_ambiguous_1700/url_fetches.jsonl"
)
DEFAULT_OUTDIR = (
    ROOT / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_semantic_repaired_ambiguous_1700"
)

COUNTRY_MARKERS = {
    "United States": ["United States", "U.S.", "US", "USA", "American"],
    "Canada": ["Canada", "Canadian"],
    "Jamaica": ["Jamaica", "Jamaican"],
    "India": ["India", "Indian"],
    "Pakistan": ["Pakistan", "Pakistani"],
    "Bangladesh": ["Bangladesh", "Bangladeshi"],
    "Sri Lanka": ["Sri Lanka", "Sri Lankan"],
    "Hong Kong": ["Hong Kong"],
    "Malaysia": ["Malaysia", "Malaysian"],
    "Philippines": ["Philippines", "Philippine", "Filipino"],
    "Nigeria": ["Nigeria", "Nigerian"],
    "South Africa": ["South Africa", "South African"],
    "Kenya": ["Kenya", "Kenyan"],
    "Ghana": ["Ghana", "Ghanaian"],
    "Tanzania": ["Tanzania", "Tanzanian"],
    "United Kingdom": ["United Kingdom", "U.K.", "UK", "Britain", "British", "England", "English"],
    "Ireland": ["Ireland", "Irish"],
}

GENERIC_SLUGS = {
    "president",
    "prime_minister",
    "parliament",
    "senate",
    "national_assembly",
    "house_of_representatives",
    "house_of_commons",
    "house_of_lords",
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

STOPWORDS = {
    "which",
    "what",
    "who",
    "where",
    "when",
    "would",
    "likely",
    "often",
    "commonly",
    "country",
    "national",
    "local",
    "news",
    "reports",
    "report",
    "article",
    "articles",
    "story",
    "coverage",
    "said",
    "says",
    "called",
    "known",
    "named",
    "used",
    "main",
    "major",
    "the",
    "and",
    "for",
    "with",
    "from",
    "about",
    "into",
    "that",
    "this",
    "was",
    "were",
    "are",
    "is",
    "as",
    "of",
    "in",
    "on",
    "to",
}


def normalize_text(text: Any) -> str:
    text = html.unescape(str(text or "")).lower()
    text = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact_text(text: str, limit: int = 120_000) -> str:
    text = re.sub(r"\s+", " ", html.unescape(str(text or ""))).strip()
    return text[:limit]


def answer_aliases(answer: str) -> set[str]:
    answer = str(answer or "").strip()
    aliases = {answer}
    aliases.add(re.sub(r"^(the|a|an)\s+", "", answer, flags=re.IGNORECASE).strip())
    aliases.add(answer.replace("&", "and"))
    aliases.add(answer.replace(" and ", " & "))
    aliases.add(answer.replace(".", ""))
    aliases.add(answer.replace("-", " "))
    return {normalize_text(alias) for alias in aliases if normalize_text(alias)}


def marker_aliases(country: str) -> set[str]:
    return {normalize_text(marker) for marker in COUNTRY_MARKERS.get(country, [country]) if normalize_text(marker)}


def contains_any(norm_text: str, aliases: set[str]) -> bool:
    padded = f" {norm_text} "
    return any(f" {alias} " in padded for alias in aliases)


def slug_from_url(url: str) -> str:
    if "/wiki/" in str(url):
        return str(url).rstrip("/").split("/wiki/")[-1].lower()
    return str(url).rstrip("/").split("/")[-1].lower()


def title_to_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe="()_,:'-&.")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def load_fetches(path: Path) -> dict[str, dict]:
    out = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                out[row["url"]] = row
    return out


def is_exact_disambiguation(fetch: dict) -> bool:
    text = normalize_text(fetch.get("text", ""))
    return (
        "this disambiguation page lists articles associated with the title" in text
        or "this disambiguation page lists articles associated with the same title" in text
    )


def side_risk(row: dict, side: str, fetches: dict[str, dict]) -> list[str]:
    answer = row.get(f"{side}_answer", "")
    url = row.get(f"{side}_evidence_url", "")
    fetch = fetches.get(url, {})
    slug = slug_from_url(url).replace("%27", "'")
    answer_slug = normalize_text(answer).replace(" ", "_")
    risks = []
    if is_exact_disambiguation(fetch):
        risks.append("disambiguation_evidence")
    if slug in GENERIC_SLUGS or answer_slug in GENERIC_SLUGS:
        risks.append("generic_evidence")
    return risks


def question_keywords(question: str, limit: int = 5) -> list[str]:
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", question) if normalize_text(w) not in STOPWORDS]
    out = []
    for word in words:
        key = word.lower()
        if key not in {w.lower() for w in out}:
            out.append(word)
    return out[:limit]


class WikiClient:
    def __init__(self, delay: float):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "LocalNewsQA-semantic-repair/1.0"})
        self.search_cache: dict[str, list[dict]] = {}
        self.page_cache: dict[str, dict] = {}
        self.last_request = 0.0

    def throttle(self) -> None:
        elapsed = time.time() - self.last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request = time.time()

    def search(self, query: str, limit: int = 8) -> list[dict]:
        if query in self.search_cache:
            return self.search_cache[query]
        response = None
        for attempt in range(4):
            self.throttle()
            response = self.session.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "srlimit": limit,
                },
                timeout=20,
            )
            if response.status_code != 429:
                break
            time.sleep(2.0 * (attempt + 1))
        response.raise_for_status()
        rows = response.json().get("query", {}).get("search", [])
        self.search_cache[query] = rows
        return rows

    def page(self, title: str) -> dict:
        if title in self.page_cache:
            return self.page_cache[title]
        response = None
        for attempt in range(4):
            self.throttle()
            response = self.session.get(
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
                    "titles": title,
                },
                timeout=25,
            )
            if response.status_code != 429:
                break
            time.sleep(2.0 * (attempt + 1))
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", [])
        page = pages[0] if pages else {"title": title, "missing": True}
        row = {
            "title": page.get("title", title),
            "fullurl": page.get("fullurl") or title_to_url(page.get("title", title)),
            "extract": compact_text(page.get("extract", "")),
            "missing": bool(page.get("missing")),
            "is_disambiguation": "disambiguation" in page.get("pageprops", {}),
        }
        self.page_cache[title] = row
        return row


def candidate_queries(row: dict, side: str) -> list[str]:
    country = row["country"] if side == "target" else row.get("contrast_country", "")
    answer = row.get(f"{side}_answer", "")
    topic = row.get("topic", "")
    keywords = " ".join(question_keywords(row.get("question", "")))
    queries = [
        f'"{answer}" "{country}"',
        f"{answer} {country} {keywords}",
        f"{answer} {country} {topic}",
    ]
    demonyms = [m for m in COUNTRY_MARKERS.get(country, []) if m != country and "." not in m and len(m) > 2]
    if demonyms:
        queries.append(f'"{answer}" "{demonyms[0]}"')
    seen = set()
    out = []
    for query in queries:
        query = re.sub(r"\s+", " ", query).strip()
        if query and query not in seen:
            out.append(query)
            seen.add(query)
    return out


def score_candidate(row: dict, side: str, search_row: dict, page: dict) -> tuple[int, list[str]]:
    country = row["country"] if side == "target" else row.get("contrast_country", "")
    answer = row.get(f"{side}_answer", "")
    title = search_row.get("title", "")
    snippet = re.sub(r"<[^>]+>", " ", html.unescape(search_row.get("snippet", "")))
    blob = " ".join([title, snippet, page.get("extract", "")])
    norm_blob = normalize_text(blob)
    norm_title = normalize_text(title)
    reasons = []
    score = 0
    if page.get("missing"):
        return -999, ["missing_page"]
    if page.get("is_disambiguation"):
        score -= 200
        reasons.append("candidate_disambiguation")
    if contains_any(norm_blob, answer_aliases(answer)):
        score += 35
        reasons.append("answer_in_candidate")
    else:
        score -= 80
        reasons.append("answer_missing")
    if contains_any(norm_blob, marker_aliases(country)):
        score += 45
        reasons.append("country_in_candidate")
    else:
        score -= 80
        reasons.append("country_missing")
    if contains_any(norm_title, answer_aliases(answer)):
        score += 20
        reasons.append("answer_in_title")
    if contains_any(norm_title, marker_aliases(country)):
        score += 30
        reasons.append("country_in_title")
    if "(disambiguation)" in title.lower() or "list of" == title.lower()[:7]:
        score -= 50
        reasons.append("bad_title_shape")
    return score, reasons


def repair_side(row: dict, side: str, fetches: dict[str, dict], wiki: WikiClient) -> tuple[dict | None, dict]:
    risks = side_risk(row, side, fetches)
    if not risks:
        return None, {"side": side, "status": "not_risky", "risks": ""}

    best = None
    attempts = []
    for query in candidate_queries(row, side):
        try:
            search_rows = wiki.search(query)
        except Exception as exc:
            attempts.append({"query": query, "error": str(exc)[:200]})
            continue
        for search_row in search_rows:
            title = search_row.get("title", "")
            try:
                page = wiki.page(title)
            except Exception as exc:
                attempts.append({"query": query, "title": title, "error": str(exc)[:200]})
                continue
            score, reasons = score_candidate(row, side, search_row, page)
            attempt = {
                "query": query,
                "title": title,
                "url": page.get("fullurl", title_to_url(title)),
                "score": score,
                "reasons": " | ".join(reasons),
            }
            attempts.append(attempt)
            if best is None or score > best["score"]:
                best = {**attempt, "page": page}

    if not best or best["score"] < 0:
        return None, {
            "side": side,
            "status": "unresolved",
            "risks": " | ".join(risks),
            "best_title": "" if not best else best["title"],
            "best_score": "" if not best else best["score"],
            "attempts_json": json.dumps(attempts[:12], ensure_ascii=False),
        }

    page = best["page"]
    repaired_fetch = {
        "url": page["fullurl"],
        "ok": True,
        "status_code": 200,
        "final_url": page["fullurl"],
        "content_type": "application/json; mediawiki extract",
        "title": page["title"],
        "text": page["extract"],
        "text_len": len(page["extract"]),
        "error": "",
        "elapsed_sec": 0.0,
        "source": "wikipedia_search_extract_semantic_repair",
    }
    return repaired_fetch, {
        "side": side,
        "status": "repaired",
        "risks": " | ".join(risks),
        "old_url": row.get(f"{side}_evidence_url", ""),
        "new_url": page["fullurl"],
        "new_title": page["title"],
        "score": best["score"],
        "score_reasons": best["reasons"],
        "attempts_json": json.dumps(attempts[:12], ensure_ascii=False),
    }


def audit_side(row: dict, side: str, fetches: dict[str, dict]) -> dict:
    country = row["country"] if side == "target" else row.get("contrast_country", "")
    answer = row.get(f"{side}_answer", "")
    url = row.get(f"{side}_evidence_url", "")
    fetch = fetches.get(url, {})
    blob = " ".join(
        [
            str(url),
            str(row.get(f"{side}_evidence_title", "")),
            str(row.get(f"{side}_evidence_excerpt", "")),
            str(fetch.get("title", "")),
            str(fetch.get("text", "")),
        ]
    )
    norm_blob = normalize_text(blob)
    return {
        f"{side}_url_ok": bool(fetch.get("ok")),
        f"{side}_answer_found": contains_any(norm_blob, answer_aliases(answer)),
        f"{side}_country_marker_found": contains_any(norm_blob, marker_aliases(country)),
        f"{side}_is_disambiguation": is_exact_disambiguation(fetch),
        f"{side}_risks": " | ".join(side_risk(row, side, fetches)),
    }


def row_semantic_status(row: dict, fetches: dict[str, dict]) -> tuple[str, list[str]]:
    reasons = []
    for side in ["target", "contrast"]:
        side_audit = audit_side(row, side, fetches)
        for key, value in side_audit.items():
            if key.endswith("_url_ok") and not value:
                reasons.append(f"{side}_url_not_ok")
            if key.endswith("_answer_found") and not value:
                reasons.append(f"{side}_answer_not_found")
            if key.endswith("_country_marker_found") and not value:
                reasons.append(f"{side}_country_marker_not_found")
            if key.endswith("_is_disambiguation") and value:
                reasons.append(f"{side}_disambiguation")
            if key.endswith("_risks") and value:
                reasons.append(f"{side}_semantic_risk:{value}")
    return ("pass" if not reasons else "fail"), reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair generic/disambiguation evidence in strict ambiguous split.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--fetch-cache", type=Path, default=DEFAULT_FETCH_CACHE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--delay", type=float, default=0.12)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    fetches = load_fetches(args.fetch_cache)
    wiki = WikiClient(delay=args.delay)
    repaired_rows = []
    repair_log = []
    new_fetch_rows = []

    for idx, row in enumerate(rows, start=1):
        out = dict(row)
        for side in ["target", "contrast"]:
            repaired_fetch, log = repair_side(out, side, fetches, wiki)
            log.update({"id": out.get("id", ""), "source_row_id": out.get("source_row_id", ""), "country": out.get("country", "")})
            repair_log.append(log)
            if repaired_fetch:
                old_url = out.get(f"{side}_evidence_url", "")
                out[f"{side}_evidence_url"] = repaired_fetch["url"]
                out[f"{side}_evidence_title"] = repaired_fetch["title"]
                out[f"{side}_evidence_excerpt"] = repaired_fetch["text"][:1000]
                out[f"{side}_evidence_final_url"] = repaired_fetch["final_url"]
                out[f"{side}_evidence_fetch_source"] = repaired_fetch["source"]
                fetches[repaired_fetch["url"]] = repaired_fetch
                new_fetch_rows.append(repaired_fetch)
                log["old_url"] = old_url
        status, reasons = row_semantic_status(out, fetches)
        out["semantic_repair_status"] = status
        out["semantic_repair_notes"] = " | ".join(reasons)
        repaired_rows.append(out)
        if idx % 100 == 0:
            print(f"processed {idx}/{len(rows)}", flush=True)

    args.outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.outdir / "localnewsqa_ambiguous_semantic_repaired_1700.jsonl"
    csv_path = args.outdir / "localnewsqa_ambiguous_semantic_repaired_1700.csv"
    repair_log_path = args.outdir / "semantic_repair_log.csv"
    new_fetches_path = args.outdir / "semantic_repair_new_fetches.jsonl"
    summary_path = args.outdir / "semantic_repair_summary.json"
    write_jsonl(jsonl_path, repaired_rows)
    write_csv(csv_path, repaired_rows)
    write_csv(repair_log_path, repair_log)
    write_jsonl(new_fetches_path, new_fetch_rows)

    status_counts = Counter(row.get("semantic_repair_status", "") for row in repaired_rows)
    log_counts = Counter(row.get("status", "") for row in repair_log)
    risk_counts = Counter(risk for row in repair_log for risk in str(row.get("risks", "")).split(" | ") if risk)
    summary = {
        "input": str(args.input),
        "rows": len(repaired_rows),
        "semantic_status_counts": dict(status_counts),
        "side_repair_status_counts": dict(log_counts),
        "side_risk_counts": dict(risk_counts),
        "rows_with_repaired_side": sum(1 for row in repaired_rows if "wikipedia_search_extract_semantic_repair" in json.dumps(row)),
        "new_fetch_rows": len(new_fetch_rows),
        "paths": {
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
            "repair_log": str(repair_log_path),
            "new_fetches": str(new_fetches_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
