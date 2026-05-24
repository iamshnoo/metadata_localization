#!/usr/bin/env python3

import argparse
import csv
import html
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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

MISSING_MARKERS = [
    "wikipedia does not have an article with this exact name",
    "the page you specified doesn't exist",
    "this page does not exist",
    "404 not found",
    "page not found",
]


def compact_text(text: Any, limit: int = 120_000) -> str:
    text = re.sub(r"\s+", " ", html.unescape(str(text or ""))).strip()
    return text[:limit]


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


def candidate_urls(
    rows: list[dict[str, Any]],
    warnings: dict[str, set[str]],
    cache: dict[str, dict[str, Any]],
    countries: set[str],
) -> list[str]:
    urls = set()
    for row in rows:
        if row.get("country") not in countries:
            continue
        row_warnings = warnings.get(row.get("source_row_id", ""), set())
        if not any(warning.startswith("target_evidence_fetch_") for warning in row_warnings):
            continue
        url = row.get("target_evidence_url", "")
        if url and not cache.get(url, {}).get("ok"):
            urls.add(url)
    return sorted(urls)


def title_from_reader_text(text: str) -> str:
    for line in text.splitlines()[:8]:
        if line.startswith("Title:"):
            return line.split(":", 1)[1].strip()
    return ""


def is_missing_page(url: str, text: str) -> bool:
    lowered = text[:20_000].lower()
    if any(marker in lowered for marker in MISSING_MARKERS):
        return True
    parsed = urlparse(url)
    if parsed.netloc.endswith("wikipedia.org") and title_from_reader_text(text).lower() in {"not found", "bad title"}:
        return True
    return False


def fetch_reader(url: str, timeout: float, retries: int) -> dict[str, Any]:
    started = time.time()
    reader_url = "https://r.jina.ai/http://" + url
    row = {
        "url": url,
        "ok": False,
        "status_code": "",
        "final_url": reader_url,
        "content_type": "text/plain; jina_reader",
        "title": "",
        "text": "",
        "text_len": 0,
        "error": "",
        "elapsed_sec": 0.0,
        "source": "jina_reader_explicit_deficit",
    }
    response = None
    for attempt in range(retries + 1):
        try:
            response = requests.get(
                reader_url,
                timeout=timeout,
                headers={"User-Agent": "LocalNewsQA-explicit-evidence-validation/1.0"},
            )
            row["status_code"] = response.status_code
            response.raise_for_status()
            text = response.text
            compact = compact_text(text)
            missing = is_missing_page(url, text)
            row["title"] = title_from_reader_text(text)
            row["text"] = compact
            row["text_len"] = len(compact)
            row["ok"] = bool(compact) and not missing
            row["error"] = "reader_detected_missing_page" if missing else ""
            break
        except Exception as exc:
            row["status_code"] = getattr(response, "status_code", "") if response is not None else ""
            row["error"] = str(exc)[:500]
            if attempt < retries:
                time.sleep(min(12.0, 2.0 * (attempt + 1)))
    row["elapsed_sec"] = round(time.time() - started, 3)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch deficit-country explicit evidence through Jina reader.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--warnings", type=Path, default=DEFAULT_WARNINGS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--quota", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=35.0)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    warnings = read_warnings(args.warnings)
    cache = load_fetch_cache(args.cache)
    countries = deficit_countries(rows, warnings, args.quota)
    urls = candidate_urls(rows, warnings, cache, countries)
    print(
        json.dumps(
            {"deficit_countries": sorted(countries), "urls": len(urls), "workers": args.workers},
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    fetched = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_url = {pool.submit(fetch_reader, url, args.timeout, args.retries): url for url in urls}
        for future in as_completed(future_to_url):
            row = future.result()
            fetched[row["url"]] = row
            done += 1
            if done % 100 == 0 or done == len(urls):
                ok = sum(1 for item in fetched.values() if item.get("ok"))
                print(f"jina fetched {done}/{len(urls)} ok={ok}", flush=True)

    cache.update(fetched)
    write_jsonl(args.cache, [cache[url] for url in sorted(cache)])
    summary = {
        "deficit_countries": sorted(countries),
        "urls": len(urls),
        "ok": sum(1 for item in fetched.values() if item.get("ok")),
        "failed": sum(1 for item in fetched.values() if not item.get("ok")),
        "failed_status_counts": dict(Counter(str(item.get("status_code", "")) for item in fetched.values() if not item.get("ok"))),
        "cache": str(args.cache),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
