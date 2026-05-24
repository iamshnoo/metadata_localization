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
from urllib.parse import quote, unquote, urlparse

import requests

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/localnewsqa_ambiguous_verifiable_1700.jsonl"
)
DEFAULT_OUTDIR = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_audit_batched_ambiguous_1700"
)

WIKIPEDIA_HOSTS = {"en.wikipedia.org", "en.m.wikipedia.org"}

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


def option_parts(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(part).strip() for part in raw if str(part).strip()]
    out = []
    for part in str(raw or "").split("||"):
        part = part.strip()
        if part:
            out.append(re.sub(r"^[A-Da-d]\s*:\s*", "", part).strip())
    return out


def answer_count(options: list[str], answer: str) -> int:
    norm = normalize_text(answer)
    return sum(1 for opt in options if normalize_text(opt) == norm)


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


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def url_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def wikipedia_title_from_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.lower() not in WIKIPEDIA_HOSTS:
        return ""
    if not parsed.path.startswith("/wiki/"):
        return ""
    return unquote(parsed.path[len("/wiki/") :]).replace("_", " ").strip()


def wikipedia_url_from_title(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe="()_,:'-")


def empty_fetch(url: str, error: str) -> dict:
    return {
        "url": url,
        "ok": False,
        "status_code": "",
        "final_url": "",
        "content_type": "",
        "title": "",
        "text": "",
        "text_len": 0,
        "error": error,
        "elapsed_sec": 0.0,
        "source": "",
    }


def fetch_wikipedia_titles(urls: list[str], batch_size: int, sleep_sec: float, retries: int) -> dict[str, dict]:
    title_to_urls: dict[str, list[str]] = {}
    for url in urls:
        title = wikipedia_title_from_url(url)
        if title:
            title_to_urls.setdefault(title, []).append(url)

    results: dict[str, dict] = {}
    titles = sorted(title_to_urls)
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "LocalNewsQA-no-key-audit/1.0 "
                "(academic dataset validation; contact: local run)"
            )
        }
    )

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
            "exintro": "1",
            "exlimit": "max",
            "titles": "|".join(batch),
        }
        response = None
        error = ""
        started = time.time()
        for attempt in range(retries + 1):
            try:
                response = session.get("https://en.wikipedia.org/w/api.php", params=params, timeout=30)
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(f"retryable status {response.status_code}", response=response)
                response.raise_for_status()
                error = ""
                break
            except Exception as exc:
                response = getattr(exc, "response", response)
                error = str(exc)[:500]
                if attempt == retries:
                    break
                time.sleep(min(8.0, 1.5 * (attempt + 1)))

        elapsed = round(time.time() - started, 3)
        if error:
            for title in batch:
                for url in title_to_urls[title]:
                    row = empty_fetch(url, error)
                    row["status_code"] = getattr(response, "status_code", "") if response is not None else ""
                    row["elapsed_sec"] = elapsed
                    row["source"] = "wikipedia_api"
                    results[url] = row
            continue

        data = response.json()
        pages = data.get("query", {}).get("pages", [])
        title_to_page = {page.get("title", ""): page for page in pages}
        for item in data.get("query", {}).get("normalized", []):
            if item.get("from") in title_to_urls and item.get("to") in title_to_page:
                title_to_page[item["from"]] = title_to_page[item["to"]]
        for item in data.get("query", {}).get("redirects", []):
            if item.get("from") in title_to_urls and item.get("to") in title_to_page:
                title_to_page[item["from"]] = title_to_page[item["to"]]

        for title in batch:
            page = title_to_page.get(title)
            if not page or page.get("missing"):
                for url in title_to_urls[title]:
                    row = empty_fetch(url, "wikipedia_page_missing")
                    row["status_code"] = 404
                    row["source"] = "wikipedia_api"
                    row["elapsed_sec"] = elapsed
                    results[url] = row
                continue

            text = compact_text(page.get("extract", ""))
            canonical = page.get("fullurl") or wikipedia_url_from_title(page.get("title", title))
            for url in title_to_urls[title]:
                results[url] = {
                    "url": url,
                    "ok": bool(text),
                    "status_code": 200,
                    "final_url": canonical,
                    "content_type": "application/json; mediawiki extract",
                    "title": page.get("title", ""),
                    "text": text,
                    "text_len": len(text),
                    "error": "" if text else "empty_wikipedia_extract",
                    "elapsed_sec": elapsed,
                    "source": "wikipedia_api",
                }

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    return results


def extract_html_text(raw: str) -> tuple[str, str]:
    if BeautifulSoup is not None:
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        text = soup.get_text(" ", strip=True)
        return title, compact_text(text)
    title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, flags=re.IGNORECASE | re.DOTALL)
    title = re.sub(r"<[^>]+>", " ", title_match.group(1)).strip() if title_match else ""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(title), compact_text(text)


def fetch_direct_url(url: str, timeout: float) -> dict:
    started = time.time()
    row = empty_fetch(url, "")
    row["source"] = "direct_http"
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "LocalNewsQA-no-key-audit/1.0"
                )
            },
            allow_redirects=True,
        )
        row["status_code"] = response.status_code
        row["final_url"] = response.url
        row["content_type"] = response.headers.get("content-type", "")
        response.raise_for_status()
        raw_text = response.text
        if "html" in row["content_type"].lower() or raw_text.lstrip().startswith("<"):
            title, text = extract_html_text(raw_text)
        else:
            title, text = "", compact_text(raw_text)
        row["title"] = title
        row["text"] = text
        row["text_len"] = len(text)
        row["ok"] = bool(text)
        row["error"] = "" if text else "empty_direct_response"
    except Exception as exc:
        row["error"] = str(exc)[:500]
    finally:
        row["elapsed_sec"] = round(time.time() - started, 3)
    return row


def fetch_all(urls: list[str], batch_size: int, sleep_sec: float, retries: int, timeout: float) -> dict[str, dict]:
    wiki_urls = [url for url in urls if wikipedia_title_from_url(url)]
    direct_urls = [url for url in urls if url and not wikipedia_title_from_url(url)]
    fetches = fetch_wikipedia_titles(wiki_urls, batch_size=batch_size, sleep_sec=sleep_sec, retries=retries)
    for url in direct_urls:
        fetches[url] = fetch_direct_url(url, timeout=timeout)
    return fetches


def audit_side(row: dict, side: str, fetches: dict[str, dict]) -> dict:
    country = row["country"] if side == "target" else row.get("contrast_country", "")
    answer = row.get("target_answer") if side == "target" else row.get("contrast_answer")
    url = row.get(f"{side}_evidence_url", "")
    supplied_title = row.get(f"{side}_evidence_title", "")
    supplied_excerpt = row.get(f"{side}_evidence_excerpt", "")
    fetch = fetches.get(url, empty_fetch(url, "not_fetched"))
    evidence_blob = " ".join(
        [
            str(url),
            str(supplied_title),
            str(supplied_excerpt),
            str(fetch.get("title", "")),
            str(fetch.get("text", "")),
        ]
    )
    norm_blob = normalize_text(evidence_blob)
    return {
        f"{side}_url_ok": bool(fetch.get("ok")),
        f"{side}_status_code": fetch.get("status_code", ""),
        f"{side}_domain": url_domain(url),
        f"{side}_source": fetch.get("source", ""),
        f"{side}_text_len": fetch.get("text_len", 0),
        f"{side}_answer_found": contains_any(norm_blob, answer_aliases(answer)),
        f"{side}_country_marker_found": contains_any(norm_blob, marker_aliases(country)),
        f"{side}_fetch_error": fetch.get("error", ""),
        f"{side}_page_title": fetch.get("title", ""),
        f"{side}_final_url": fetch.get("final_url", ""),
    }


def audit_row(row: dict, fetches: dict[str, dict]) -> dict:
    options = option_parts(row.get("options", []))
    target = row.get("target_answer", "")
    contrast = row.get("contrast_answer", "")
    question_norm = normalize_text(row.get("question", ""))
    target_markers = marker_aliases(row.get("country", ""))
    contrast_markers = marker_aliases(row.get("contrast_country", ""))
    target_side = audit_side(row, "target", fetches)
    contrast_side = audit_side(row, "contrast", fetches)
    failures = []
    warnings = []

    if len(options) != 4:
        failures.append("option_count_not_4")
    if answer_count(options, target) != 1:
        failures.append("target_answer_not_exactly_once")
    if answer_count(options, contrast) < 1:
        failures.append("contrast_answer_absent")
    if contains_any(question_norm, target_markers):
        failures.append("target_country_marker_in_question")
    if contains_any(question_norm, contrast_markers):
        failures.append("contrast_country_marker_in_question")

    for side, side_result in [("target", target_side), ("contrast", contrast_side)]:
        if not side_result[f"{side}_url_ok"]:
            failures.append(f"{side}_url_fetch_failed")
        if side_result[f"{side}_url_ok"] and not side_result[f"{side}_answer_found"]:
            failures.append(f"{side}_answer_not_found_in_evidence")
        if side_result[f"{side}_url_ok"] and not side_result[f"{side}_country_marker_found"]:
            warnings.append(f"{side}_country_marker_not_found_in_evidence")
        if side_result.get(f"{side}_text_len", 0) and side_result[f"{side}_text_len"] < 500:
            warnings.append(f"{side}_evidence_text_short")

    severity = "pass"
    if failures:
        severity = "fail"
    elif warnings:
        severity = "warn"
    return {
        "id": row.get("id", ""),
        "source_row_id": row.get("source_row_id", ""),
        "country": row.get("country", ""),
        "contrast_country": row.get("contrast_country", ""),
        "topic": row.get("topic", ""),
        "year": row.get("year", ""),
        "question": row.get("question", ""),
        "target_answer": target,
        "contrast_answer": contrast,
        "weak_locale_risk": row.get("weak_locale_risk", ""),
        "llm_accept_or_restored": row.get("llm_accept_or_restored", ""),
        "severity": severity,
        "failures": " | ".join(failures),
        "warnings": " | ".join(warnings),
        **target_side,
        **contrast_side,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="No-key batched web audit for LocalNewsQA evidence URLs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    urls = sorted(
        {
            url
            for row in rows
            for url in [row.get("target_evidence_url", ""), row.get("contrast_evidence_url", "")]
            if url
        }
    )
    fetches = fetch_all(
        urls,
        batch_size=args.batch_size,
        sleep_sec=args.sleep,
        retries=args.retries,
        timeout=args.timeout,
    )
    audits = [audit_row(row, fetches) for row in rows]
    audits.sort(key=lambda r: ({"fail": 0, "warn": 1, "pass": 2}[r["severity"]], r["country"], r["id"]))

    fail_rows = [row for row in audits if row["severity"] == "fail"]
    warn_rows = [row for row in audits if row["severity"] == "warn"]
    pass_rows = [row for row in audits if row["severity"] == "pass"]

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.outdir / "url_fetches.jsonl", [fetches[url] for url in urls])
    write_csv(args.outdir / "row_web_audit.csv", audits)
    write_csv(args.outdir / "fail_rows.csv", fail_rows)
    write_csv(args.outdir / "warn_rows.csv", warn_rows)
    write_csv(args.outdir / "pass_rows.csv", pass_rows)

    summary = {
        "input": str(args.input),
        "rows": len(rows),
        "unique_urls": len(urls),
        "fetch_ok": sum(1 for url in urls if fetches.get(url, {}).get("ok")),
        "fetch_failed": sum(1 for url in urls if not fetches.get(url, {}).get("ok")),
        "fetch_source_counts": dict(Counter(fetches[url].get("source", "") for url in urls)),
        "failed_fetch_status_counts": dict(
            Counter(str(fetches[url].get("status_code", "")) for url in urls if not fetches.get(url, {}).get("ok"))
        ),
        "severity_counts": dict(Counter(row["severity"] for row in audits)),
        "failure_counts": dict(Counter(mode for row in fail_rows for mode in row["failures"].split(" | ") if mode)),
        "warning_counts": dict(Counter(mode for row in warn_rows + fail_rows for mode in row["warnings"].split(" | ") if mode)),
        "fail_by_country": dict(Counter(row["country"] for row in fail_rows)),
        "warn_by_country": dict(Counter(row["country"] for row in warn_rows)),
        "paths": {
            "row_web_audit": str(args.outdir / "row_web_audit.csv"),
            "fail_rows": str(args.outdir / "fail_rows.csv"),
            "warn_rows": str(args.outdir / "warn_rows.csv"),
            "pass_rows": str(args.outdir / "pass_rows.csv"),
            "url_fetches": str(args.outdir / "url_fetches.jsonl"),
            "summary": str(args.outdir / "summary.json"),
        },
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
