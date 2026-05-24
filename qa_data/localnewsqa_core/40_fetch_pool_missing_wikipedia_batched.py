#!/usr/bin/env python3

import argparse
import html
import json
import re
import time
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import requests


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/localnewsqa_ambiguous_verifiable_pool_4625.jsonl"
)
DEFAULT_CACHE = (
    ROOT / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_audit_ambiguous_1700/url_fetches.jsonl"
)
WIKI_HOSTS = {"en.wikipedia.org", "en.m.wikipedia.org"}


def compact_text(text: str, limit: int = 120_000) -> str:
    text = re.sub(r"\s+", " ", html.unescape(str(text or ""))).strip()
    return text[:limit]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def wiki_title(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.lower() not in WIKI_HOSTS or not parsed.path.startswith("/wiki/"):
        return ""
    return unquote(parsed.path[len("/wiki/") :]).replace("_", " ").strip()


def title_to_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe="()_,:'-&.")


def empty_fetch(url: str, error: str, status_code: str | int = "") -> dict:
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
        "source": "wikipedia_batch_extract",
    }


def needed_urls(rows: list[dict], cache: dict[str, dict], countries: set[str], retry_failed: bool) -> list[str]:
    urls = set()
    for row in rows:
        if countries and row.get("country") not in countries:
            continue
        for url in [row.get("target_evidence_url", ""), row.get("contrast_evidence_url", "")]:
            if not url:
                continue
            cached = cache.get(url)
            if cached is None or (retry_failed and not cached.get("ok")):
                urls.add(url)
    return sorted(urls)


def fetch_wiki_batches(urls: list[str], batch_size: int, sleep: float, retries: int) -> dict[str, dict]:
    title_to_urls: dict[str, list[str]] = {}
    for url in urls:
        title = wiki_title(url)
        if title:
            title_to_urls.setdefault(title, []).append(url)
    titles = sorted(title_to_urls)
    out = {}
    session = requests.Session()
    session.headers.update({"User-Agent": "LocalNewsQA-known-title-batch-fetch/1.0"})
    for start in range(0, len(titles), batch_size):
        batch = titles[start : start + batch_size]
        response = None
        error = ""
        started = time.time()
        for attempt in range(retries + 1):
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
                if attempt < retries:
                    time.sleep(2.0 * (attempt + 1))
        elapsed = round(time.time() - started, 3)
        if error:
            for title in batch:
                for url in title_to_urls[title]:
                    out[url] = empty_fetch(url, error, getattr(response, "status_code", ""))
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
                    out[url] = empty_fetch(url, "wikipedia_page_missing", 404)
                continue
            text = compact_text(page.get("extract", ""))
            final_url = page.get("fullurl") or title_to_url(page.get("title", title))
            is_disambiguation = "disambiguation" in page.get("pageprops", {})
            if is_disambiguation:
                text = f"{text} This disambiguation page lists articles associated with the title {page.get('title', title)}."
            for url in title_to_urls[title]:
                out[url] = {
                    "url": url,
                    "ok": bool(text),
                    "status_code": 200,
                    "final_url": final_url,
                    "content_type": "application/json; mediawiki extract",
                    "title": page.get("title", ""),
                    "text": text,
                    "text_len": len(text),
                    "error": "" if text else "empty_wikipedia_extract",
                    "elapsed_sec": elapsed,
                    "source": "wikipedia_batch_extract",
                }
        print(f"wiki batch {start + len(batch)}/{len(titles)}", flush=True)
        if sleep:
            time.sleep(sleep)
    return out


def fetch_direct(url: str, timeout: float) -> dict:
    started = time.time()
    row = empty_fetch(url, "", "")
    row["source"] = "direct_http_batch_fallback"
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "LocalNewsQA-known-title-batch-fetch/1.0"})
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
    parser = argparse.ArgumentParser(description="Batch-fetch missing pool evidence URLs by known Wikipedia title.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--countries", nargs="*", default=[])
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    cache = load_fetches(args.cache)
    urls = needed_urls(rows, cache, set(args.countries), args.retry_failed)
    wiki_urls = [url for url in urls if wiki_title(url)]
    direct_urls = [url for url in urls if not wiki_title(url)]
    fetched = fetch_wiki_batches(wiki_urls, args.batch_size, args.sleep, args.retries)
    for idx, url in enumerate(direct_urls, start=1):
        fetched[url] = fetch_direct(url, args.timeout)
        print(f"direct {idx}/{len(direct_urls)}", flush=True)
        if args.sleep:
            time.sleep(args.sleep)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    with args.cache.open("a", encoding="utf-8") as handle:
        for url in urls:
            if url in fetched:
                handle.write(json.dumps(fetched[url], ensure_ascii=False) + "\n")
    summary = {
        "countries": args.countries,
        "urls": len(urls),
        "wiki_urls": len(wiki_urls),
        "direct_urls": len(direct_urls),
        "ok": sum(1 for row in fetched.values() if row.get("ok")),
        "failed": sum(1 for row in fetched.values() if not row.get("ok")),
        "cache": str(args.cache),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
