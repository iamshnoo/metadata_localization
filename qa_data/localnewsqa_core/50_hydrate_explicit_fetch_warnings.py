#!/usr/bin/env python3

import argparse
import csv
import html
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import requests


ROOT = Path(__file__).resolve().parents[2]
STRICT_DIR = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_semantic_gold_ambiguous_1700"
)
AUDIT_DIR = STRICT_DIR / "explicit_max_audit"
DEFAULT_INPUT = AUDIT_DIR / "localnewsqa_targetqa_explicit_style_max_paper_clean.jsonl"
DEFAULT_WARNINGS = AUDIT_DIR / "explicit_max_quality_warnings.csv"
DEFAULT_CACHE = AUDIT_DIR / "explicit_target_evidence_fetches.jsonl"
WIKI_HOSTS = {"en.wikipedia.org", "en.m.wikipedia.org"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact_text(text: Any, limit: int = 120_000) -> str:
    text = re.sub(r"\s+", " ", html.unescape(str(text or ""))).strip()
    return text[:limit]


def wiki_title(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.lower() not in WIKI_HOSTS or not parsed.path.startswith("/wiki/"):
        return ""
    return unquote(parsed.path[len("/wiki/") :]).replace("_", " ").strip()


def empty_fetch(url: str, error: str, status_code: str | int = "", source: str = "explicit_warning_hydration") -> dict[str, Any]:
    return {
        "url": url,
        "ok": False,
        "status_code": status_code,
        "final_url": "",
        "content_type": "",
        "title": "",
        "text": "",
        "text_len": 0,
        "error": error,
        "elapsed_sec": 0.0,
        "source": source,
    }


def read_warnings(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    if not path.exists() or path.stat().st_size == 0:
        return out
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source_id = row.get("source_row_id", "")
            warnings = {part.strip() for part in row.get("warnings", "").split("|") if part.strip()}
            if source_id:
                out[source_id] = warnings
    return out


def deficit_countries(rows: list[dict[str, Any]], warnings: dict[str, set[str]], quota: int) -> set[str]:
    countries = sorted({row.get("country", "") for row in rows})
    out = set()
    for country in countries:
        strict_count = sum(
            1
            for row in rows
            if row.get("country") == country and not warnings.get(row.get("source_row_id", ""), set())
        )
        if strict_count < quota:
            out.add(country)
    return out


def needed_urls(
    rows: list[dict[str, Any]],
    warnings: dict[str, set[str]],
    cache: dict[str, dict[str, Any]],
    countries: set[str],
    all_countries: bool,
) -> list[str]:
    urls = set()
    for row in rows:
        if countries and row.get("country") not in countries and not all_countries:
            continue
        row_warnings = warnings.get(row.get("source_row_id", ""), set())
        if not any(warning.startswith("target_evidence_fetch_") for warning in row_warnings):
            continue
        url = row.get("target_evidence_url", "")
        if not url:
            continue
        cached = cache.get(url, {})
        if not cached.get("ok"):
            urls.add(url)
    return sorted(urls)


def fetch_wikipedia_batches(urls: list[str], batch_size: int, sleep: float, retries: int) -> dict[str, dict[str, Any]]:
    title_to_urls: dict[str, list[str]] = defaultdict(list)
    for url in urls:
        title = wiki_title(url)
        if title:
            title_to_urls[title].append(url)
    titles = sorted(title_to_urls)
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "LocalNewsQA-explicit-evidence-validation/1.0 "
                "(academic dataset validation; no paid API)"
            )
        }
    )
    out = {}
    for start in range(0, len(titles), batch_size):
        batch = titles[start : start + batch_size]
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "redirects": "1",
            "prop": "extracts|info",
            "inprop": "url",
            "explaintext": "1",
            "exlimit": "max",
            "titles": "|".join(batch),
        }
        response = None
        error = ""
        started = time.time()
        for attempt in range(retries + 1):
            try:
                response = session.get("https://en.wikipedia.org/w/api.php", params=params, timeout=40)
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(f"retryable status {response.status_code}", response=response)
                response.raise_for_status()
                error = ""
                break
            except Exception as exc:
                response = getattr(exc, "response", response)
                error = str(exc)[:500]
                if attempt < retries:
                    retry_after = 0
                    if response is not None:
                        retry_after = int(response.headers.get("retry-after", "0") or 0)
                    time.sleep(max(retry_after, min(60.0, 2.0 * (attempt + 1) ** 2)))
        elapsed = round(time.time() - started, 3)
        if error:
            status_code = getattr(response, "status_code", "") if response is not None else ""
            for title in batch:
                for url in title_to_urls[title]:
                    row = empty_fetch(url, error, status_code)
                    row["elapsed_sec"] = elapsed
                    out[url] = row
            print(f"wiki batch failed {start + len(batch)}/{len(titles)} status={status_code}", flush=True)
            if sleep:
                time.sleep(sleep)
            continue

        data = response.json()
        pages = data.get("query", {}).get("pages", [])
        page_by_title = {page.get("title", ""): page for page in pages}
        for item in data.get("query", {}).get("normalized", []):
            if item.get("from") in title_to_urls and item.get("to") in page_by_title:
                page_by_title[item["from"]] = page_by_title[item["to"]]
        for item in data.get("query", {}).get("redirects", []):
            if item.get("from") in title_to_urls and item.get("to") in page_by_title:
                page_by_title[item["from"]] = page_by_title[item["to"]]

        for title in batch:
            page = page_by_title.get(title)
            if not page or page.get("missing"):
                for url in title_to_urls[title]:
                    row = empty_fetch(url, "wikipedia_page_missing", 404)
                    row["elapsed_sec"] = elapsed
                    out[url] = row
                continue
            text = compact_text(page.get("extract", ""))
            for url in title_to_urls[title]:
                out[url] = {
                    "url": url,
                    "ok": bool(text),
                    "status_code": 200,
                    "final_url": page.get("fullurl", url),
                    "content_type": "application/json; mediawiki full extract",
                    "title": page.get("title", title),
                    "text": text,
                    "text_len": len(text),
                    "error": "" if text else "empty_wikipedia_extract",
                    "elapsed_sec": elapsed,
                    "source": "explicit_warning_hydration_wikipedia_api",
                }
        print(f"wiki hydrated {start + len(batch)}/{len(titles)}", flush=True)
        if sleep:
            time.sleep(sleep)
    return out


def fetch_direct(url: str, timeout: float) -> dict[str, Any]:
    started = time.time()
    row = empty_fetch(url, "", "", "explicit_warning_hydration_direct_http")
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "LocalNewsQA-explicit-evidence-validation/1.0"},
        )
        row["status_code"] = response.status_code
        row["final_url"] = response.url
        row["content_type"] = response.headers.get("content-type", "")
        response.raise_for_status()
        row["text"] = compact_text(response.text)
        row["text_len"] = len(row["text"])
        row["ok"] = bool(row["text"])
    except Exception as exc:
        row["error"] = str(exc)[:500]
    row["elapsed_sec"] = round(time.time() - started, 3)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Hydrate fetch-warning explicit evidence rows without paid APIs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--warnings", type=Path, default=DEFAULT_WARNINGS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--quota", type=int, default=1000)
    parser.add_argument("--all-countries", action="store_true")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--sleep", type=float, default=0.35)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=25.0)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    warnings = read_warnings(args.warnings)
    cache = load_fetch_cache(args.cache)
    countries = deficit_countries(rows, warnings, args.quota)
    urls = needed_urls(rows, warnings, cache, countries, args.all_countries)
    wiki_urls = [url for url in urls if wiki_title(url)]
    direct_urls = [url for url in urls if not wiki_title(url)]
    print(
        json.dumps(
            {
                "deficit_countries": sorted(countries),
                "needed_urls": len(urls),
                "wiki_urls": len(wiki_urls),
                "direct_urls": len(direct_urls),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    fetched = fetch_wikipedia_batches(wiki_urls, args.batch_size, args.sleep, args.retries)
    for index, url in enumerate(direct_urls, start=1):
        fetched[url] = fetch_direct(url, args.timeout)
        print(f"direct hydrated {index}/{len(direct_urls)}", flush=True)
        if args.sleep:
            time.sleep(args.sleep)

    cache.update(fetched)
    write_jsonl(args.cache, [cache[url] for url in sorted(cache)])
    summary = {
        "deficit_countries": sorted(countries),
        "urls": len(urls),
        "wiki_urls": len(wiki_urls),
        "direct_urls": len(direct_urls),
        "fetched_ok": sum(1 for row in fetched.values() if row.get("ok")),
        "fetched_failed": sum(1 for row in fetched.values() if not row.get("ok")),
        "failed_status_counts": dict(Counter(str(row.get("status_code", "")) for row in fetched.values() if not row.get("ok"))),
        "cache": str(args.cache),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
