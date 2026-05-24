#!/usr/bin/env python3

import argparse
import html
import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/localnewsqa_ambiguous_verifiable_1700.jsonl"
)
DEFAULT_CACHE = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_audit_ambiguous_1700/url_fetches.jsonl"
)


MISSING_MARKERS = [
    "wikipedia does not have an article with this exact name",
    "the page you specified doesn't exist",
    "this page does not exist",
    "404 not found",
    "page not found",
]


def compact_text(text: str, limit: int = 120_000) -> str:
    text = re.sub(r"\s+", " ", html.unescape(str(text or ""))).strip()
    return text[:limit]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_last_fetches(path: Path) -> dict[str, dict]:
    out = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                out[row["url"]] = row
    return out


def candidate_urls(input_path: Path, cache_path: Path, mode: str, country: str) -> list[str]:
    rows = read_jsonl(input_path)
    if country:
        rows = [row for row in rows if row.get("country") == country]
    urls = sorted(
        {
            url
            for row in rows
            for url in [row.get("target_evidence_url", ""), row.get("contrast_evidence_url", "")]
            if url
        }
    )
    fetches = load_last_fetches(cache_path)
    if mode == "all":
        return urls
    if mode == "missing":
        return [url for url in urls if url not in fetches]
    if mode == "failed":
        return [url for url in urls if url not in fetches or not fetches[url].get("ok")]
    raise ValueError(f"unknown mode: {mode}")


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


def fetch_reader(url: str, timeout: float, retries: int) -> dict:
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
        "source": "jina_reader",
    }
    response = None
    for attempt in range(retries + 1):
        try:
            response = requests.get(
                reader_url,
                timeout=timeout,
                headers={"User-Agent": "LocalNewsQA-no-key-audit/1.0"},
            )
            row["status_code"] = response.status_code
            response.raise_for_status()
            text = response.text
            title = title_from_reader_text(text)
            compact = compact_text(text)
            missing = is_missing_page(url, text)
            row["title"] = title
            row["text"] = compact
            row["text_len"] = len(compact)
            row["ok"] = bool(compact) and not missing
            row["error"] = "reader_detected_missing_page" if missing else ""
            break
        except Exception as exc:
            row["status_code"] = getattr(response, "status_code", "") if response is not None else ""
            row["error"] = str(exc)[:500]
            if attempt < retries:
                time.sleep(min(8.0, 1.5 * (attempt + 1)))
    row["elapsed_sec"] = round(time.time() - started, 3)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch failed evidence URLs through a no-key public page reader.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--mode", choices=["failed", "missing", "all"], default="failed")
    parser.add_argument("--country", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    urls = candidate_urls(args.input, args.cache, args.mode, args.country)
    if args.limit:
        urls = urls[: args.limit]

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    counts = {"ok": 0, "failed": 0}
    with args.cache.open("a", encoding="utf-8") as handle:
        for idx, url in enumerate(urls, start=1):
            row = fetch_reader(url, timeout=args.timeout, retries=args.retries)
            counts["ok" if row.get("ok") else "failed"] += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{idx}/{len(urls)} ok={row.get('ok')} status={row.get('status_code')} {url}", flush=True)
            if args.delay > 0 and idx < len(urls):
                time.sleep(args.delay)

    print(json.dumps({"input": str(args.input), "cache": str(args.cache), "urls": len(urls), **counts}, indent=2))


if __name__ == "__main__":
    main()
