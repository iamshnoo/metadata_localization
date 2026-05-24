#!/usr/bin/env python3

import argparse
import csv
import html
import json
import re
import ssl
import time
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlencode, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/localnewsqa_ambiguous_verifiable_1700.jsonl"
)
DEFAULT_OUTDIR = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_audit_ambiguous_1700"
)

COUNTRY_MARKERS = {
    "United States": ["united states", "u.s.", "us ", "usa", "american", "america"],
    "Canada": ["canada", "canadian"],
    "Jamaica": ["jamaica", "jamaican"],
    "India": ["india", "indian"],
    "Pakistan": ["pakistan", "pakistani"],
    "Bangladesh": ["bangladesh", "bangladeshi"],
    "Sri Lanka": ["sri lanka", "sri lankan"],
    "Hong Kong": ["hong kong"],
    "Malaysia": ["malaysia", "malaysian"],
    "Philippines": ["philippines", "philippine", "filipino"],
    "Nigeria": ["nigeria", "nigerian"],
    "South Africa": ["south africa", "south african"],
    "Kenya": ["kenya", "kenyan"],
    "Ghana": ["ghana", "ghanaian"],
    "Tanzania": ["tanzania", "tanzanian"],
    "United Kingdom": ["united kingdom", "u.k.", "uk ", "britain", "british", "england", "english"],
    "Ireland": ["ireland", "irish"],
}

AUTHORITATIVE_HINTS = {
    ".gov",
    ".edu",
    ".ac.",
    ".org",
    "parliament",
    "government",
    "ministry",
    "commission",
    "authority",
    "regulator",
    "bbc.",
}


def normalize(text: str) -> str:
    text = html.unescape(str(text or "")).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact_text(text: str, limit: int = 250_000) -> str:
    text = re.sub(r"(?is)<script.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def option_parts(raw: list | str) -> list[str]:
    if isinstance(raw, list):
        return [str(part).strip() for part in raw if str(part).strip()]
    out = []
    for part in str(raw or "").split("||"):
        part = part.strip()
        if part:
            out.append(re.sub(r"^[A-Da-d]\s*:\s*", "", part).strip())
    return out


def answer_aliases(answer: str) -> set[str]:
    answer = str(answer or "").strip()
    aliases = {answer}
    aliases.add(answer.replace("&", "and"))
    aliases.add(answer.replace("and", "&"))
    aliases.add(answer.replace(".", ""))
    aliases.add(answer.replace("'", ""))
    if ":" in answer:
        aliases.add(answer.split(":", 1)[0].strip())
    if "(" in answer:
        aliases.add(re.sub(r"\s*\([^)]*\)", "", answer).strip())
    return {normalize(alias) for alias in aliases if normalize(alias)}


def contains_any(text_norm: str, needles: set[str] | list[str]) -> bool:
    padded = f" {text_norm} "
    for needle in needles:
        n = normalize(needle)
        if n and f" {n} " in padded:
            return True
    return False


def fetch_url(url: str, timeout: float) -> dict:
    if not url:
        return {"url": url, "status": "missing_url", "http_status": "", "text": "", "final_url": "", "error": "missing URL"}
    parsed = urlparse(url)
    if parsed.netloc.lower() in {"en.wikipedia.org", "en.m.wikipedia.org"} and parsed.path.startswith("/wiki/"):
        return fetch_wikipedia_raw_page(url, timeout)

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; LocalNewsQA-NoAPI-Audit/1.0; +https://example.org/audit)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7",
    }
    req = Request(iri_to_uri(url), headers=headers)
    ctx = ssl.create_default_context()
    try:
        with urlopen(req, timeout=timeout, context=ctx) as resp:
            raw = resp.read(1_500_000)
            charset = resp.headers.get_content_charset() or "utf-8"
            text = raw.decode(charset, errors="replace")
            return {
                "url": url,
                "status": "ok",
                "http_status": getattr(resp, "status", ""),
                "final_url": resp.geturl(),
                "content_type": resp.headers.get("content-type", ""),
                "text": compact_text(text),
                "error": "",
            }
    except HTTPError as exc:
        body = ""
        try:
            body = exc.read(300_000).decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return {
            "url": url,
            "status": "http_error",
            "http_status": exc.code,
            "final_url": exc.geturl() if hasattr(exc, "geturl") else url,
            "content_type": getattr(exc, "headers", {}).get("content-type", "") if getattr(exc, "headers", None) else "",
            "text": compact_text(body),
            "error": str(exc),
        }
    except (URLError, TimeoutError, ssl.SSLError, ValueError, OSError) as exc:
        return {"url": url, "status": "fetch_error", "http_status": "", "final_url": "", "content_type": "", "text": "", "error": repr(exc)}


def wikipedia_title(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.lower() not in {"en.wikipedia.org", "en.m.wikipedia.org"}:
        return ""
    if not parsed.path.startswith("/wiki/"):
        return ""
    return unquote(parsed.path.removeprefix("/wiki/")).split("#", 1)[0]


def is_wikipedia_url(url: str) -> bool:
    return bool(wikipedia_title(url))


def fetch_wikipedia_batches(
    urls: list[str], timeout: float, batch_size: int, sleep_seconds: float
) -> dict[str, dict]:
    title_to_urls: dict[str, list[str]] = {}
    for url in urls:
        title = wikipedia_title(url)
        if title:
            title_to_urls.setdefault(title, []).append(url)

    titles = sorted(title_to_urls)
    title_results: dict[str, dict] = {}
    for start in range(0, len(titles), batch_size):
        batch = titles[start : start + batch_size]
        title_results.update(fetch_wikipedia_query_batch(batch, timeout))
        print(
            json.dumps(
                {
                    "wiki_titles_fetched": min(start + len(batch), len(titles)),
                    "total_wiki_titles": len(titles),
                }
            ),
            flush=True,
        )
        if sleep_seconds > 0 and start + batch_size < len(titles):
            time.sleep(sleep_seconds)

    redirect_results = fetch_redirect_targets(title_results, timeout, batch_size, sleep_seconds)
    all_results = {**title_results, **redirect_results}
    result_by_norm = {normalize_wiki_title(title): data for title, data in all_results.items()}

    fetched_by_url: dict[str, dict] = {}
    for title in titles:
        data = title_results.get(title) or {
            "url": "",
            "status": "fetch_error",
            "http_status": "",
            "final_url": "",
            "content_type": "",
            "text": "",
            "error": "title missing from batch result",
        }
        data = merge_redirect_chain(data, result_by_norm)
        for original_url in title_to_urls[title]:
            fetched_by_url[original_url] = {**data, "url": original_url}
    return fetched_by_url


def clean_redirect_title(raw: str) -> str:
    title = str(raw or "").split("|", 1)[0].split("#", 1)[0].strip()
    return title.replace(" ", "_")


def redirect_target(data: dict) -> str:
    return clean_redirect_title(data.get("redirect_target", ""))


def fetch_redirect_targets(
    initial_results: dict[str, dict], timeout: float, batch_size: int, sleep_seconds: float
) -> dict[str, dict]:
    fetched: dict[str, dict] = {}
    known_titles = {clean_redirect_title(title) for title in initial_results}
    current = initial_results
    for depth in range(3):
        pending = sorted(
            {
                target
                for data in current.values()
                for target in [redirect_target(data)]
                if target and target not in known_titles
            }
        )
        if not pending:
            break
        next_results: dict[str, dict] = {}
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            next_results.update(fetch_wikipedia_query_batch(batch, timeout))
            print(
                json.dumps(
                    {
                        "wiki_redirect_depth": depth + 1,
                        "redirect_titles_fetched": min(start + len(batch), len(pending)),
                        "total_redirect_titles": len(pending),
                    }
                ),
                flush=True,
            )
            if sleep_seconds > 0 and start + batch_size < len(pending):
                time.sleep(sleep_seconds)
        fetched.update(next_results)
        known_titles.update(clean_redirect_title(title) for title in next_results)
        current = next_results
    return fetched


def merge_redirect_chain(data: dict, result_by_norm: dict[str, dict]) -> dict:
    merged = dict(data)
    seen = set()
    current = data
    for _ in range(3):
        target = redirect_target(current)
        target_norm = normalize_wiki_title(target)
        if not target or target_norm in seen:
            break
        seen.add(target_norm)
        target_data = result_by_norm.get(target_norm)
        if not target_data or target_data.get("status") != "ok":
            break
        merged["text"] = " ".join(
            part for part in [merged.get("text", ""), target_data.get("text", "")] if part
        )
        merged["final_url"] = target_data.get("final_url", merged.get("final_url", ""))
        current = target_data
    return merged


def fetch_wikipedia_query_batch(titles: list[str], timeout: float) -> dict[str, dict]:
    if not titles:
        return {}
    export_url = "https://en.wikipedia.org/wiki/Special:Export"
    body = urlencode({"pages": "\n".join(titles), "curonly": "1"}).encode()
    req = Request(
        export_url,
        data=body,
        headers={
            "User-Agent": "LocalNewsQA-NoAPI-Audit/1.0 (research evidence URL validation; contact: amukher6@umd.edu)",
            "Accept": "application/xml,text/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            text = raw.decode("utf-8", errors="replace")
            return parse_wikipedia_export_response(
                titles,
                text,
                getattr(resp, "status", ""),
                resp.headers.get("content-type", ""),
            )
    except HTTPError as exc:
        body = ""
        try:
            body = exc.read(300_000).decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return {
            title: {
                "url": "",
                "status": "http_error",
                "http_status": exc.code,
                "final_url": export_url,
                "content_type": getattr(exc, "headers", {}).get("content-type", "") if getattr(exc, "headers", None) else "",
                "text": compact_text(body),
                "error": str(exc),
            }
            for title in titles
        }
    except (URLError, TimeoutError, ssl.SSLError, ValueError, OSError, json.JSONDecodeError) as exc:
        return {
            title: {
                "url": "",
                "status": "fetch_error",
                "http_status": "",
                "final_url": export_url,
                "content_type": "",
                "text": "",
                "error": repr(exc),
            }
            for title in titles
        }


def normalize_wiki_title(title: str) -> str:
    return normalize(str(title or "").replace("_", " "))


def parse_wikipedia_export_response(
    titles: list[str], xml_text: str, http_status: int | str, content_type: str
) -> dict[str, dict]:
    namespace = {"mw": "http://www.mediawiki.org/xml/export-0.11/"}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        return {
            title: {
                "url": "",
                "status": "fetch_error",
                "http_status": http_status,
                "final_url": "https://en.wikipedia.org/wiki/Special:Export",
                "content_type": content_type,
                "text": "",
                "error": f"XML parse error: {exc}",
            }
            for title in titles
        }

    pages_by_title = {}
    for page in root.findall("mw:page", namespace):
        page_title = page.findtext("mw:title", default="", namespaces=namespace)
        page_text = page.findtext("mw:revision/mw:text", default="", namespaces=namespace) or ""
        pages_by_title[normalize_wiki_title(page_title)] = {
            "title": page_title,
            "text": page_text,
        }

    out = {}
    for original in titles:
        page = pages_by_title.get(normalize_wiki_title(original)) or {}
        display_title = page.get("title") or original.replace("_", " ")
        if not page:
            out[original] = {
                "url": "",
                "status": "http_error",
                "http_status": 404,
                "final_url": "https://en.wikipedia.org/wiki/" + quote(original.replace(" ", "_"), safe="/:%"),
                "content_type": content_type,
                "text": "",
                "error": "Wikipedia page missing",
                "redirect_target": "",
            }
            continue
        page_text = page.get("text", "")
        redirect_match = re.search(r"(?im)^#redirect\s*\[\[([^\]]+)\]\]", page_text)
        redirect_target = redirect_match.group(1) if redirect_match else ""
        text = " ".join(
            part
            for part in [
                original.replace("_", " "),
                display_title,
                redirect_target,
                "https://en.wikipedia.org/wiki/" + quote(display_title.replace(" ", "_"), safe="/:%"),
                page_text,
            ]
            if part
        )
        out[original] = {
            "url": "",
            "status": "ok",
            "http_status": http_status,
            "final_url": "https://en.wikipedia.org/wiki/" + quote(display_title.replace(" ", "_"), safe="/:%"),
            "content_type": content_type,
            "text": compact_text(text),
            "error": "",
            "redirect_target": redirect_target,
        }
    return out


def iri_to_uri(url: str) -> str:
    parsed = urlparse(url)
    path = quote(unquote(parsed.path), safe="/:%")
    query = quote(unquote(parsed.query), safe="=&?/:+,%")
    out = parsed._replace(path=path, query=query)
    return out.geturl()


def fetch_wikipedia_raw_page(url: str, timeout: float) -> dict:
    parsed = urlparse(url)
    title = unquote(parsed.path.removeprefix("/wiki/")).split("#", 1)[0]
    if not title:
        return {"url": url, "status": "missing_url", "http_status": "", "text": "", "final_url": "", "error": "missing Wikipedia page title"}
    raw_url = "https://en.wikipedia.org/w/index.php?title=" + quote(title, safe="") + "&action=raw"
    req = Request(
        raw_url,
        headers={
            "User-Agent": "LocalNewsQA-NoAPI-Audit/1.0 (research evidence URL validation; contact: amukher6@umd.edu)",
            "Accept": "text/plain,text/*;q=0.9,*/*;q=0.7",
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read(1_500_000)
            text = raw.decode("utf-8", errors="replace")
            redirect_match = re.search(r"(?im)^#redirect\s*\[\[([^\]]+)\]\]", text)
            if redirect_match:
                text += " " + redirect_match.group(1)
            return {
                "url": url,
                "status": "ok",
                "http_status": getattr(resp, "status", ""),
                "final_url": "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe="/:%"),
                "content_type": resp.headers.get("content-type", ""),
                "text": compact_text(text),
                "error": "",
            }
    except HTTPError as exc:
        body = ""
        try:
            body = exc.read(100_000).decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return {
            "url": url,
            "status": "http_error",
            "http_status": exc.code,
            "final_url": raw_url,
            "content_type": getattr(exc, "headers", {}).get("content-type", "") if getattr(exc, "headers", None) else "",
            "text": compact_text(body),
            "error": str(exc),
        }
    except (URLError, TimeoutError, ssl.SSLError, ValueError, OSError) as exc:
        return {"url": url, "status": "fetch_error", "http_status": "", "final_url": raw_url, "content_type": "", "text": "", "error": repr(exc)}


def domain_authority(url: str) -> str:
    host = urlparse(url or "").netloc.lower()
    whole = (host + " " + (url or "").lower())
    if "wikipedia.org" in host:
        return "wikipedia"
    if any(hint in whole for hint in AUTHORITATIVE_HINTS):
        return "authoritative_hint"
    return "ordinary_web"


def side_audit(row: dict, side: str, fetched: dict) -> dict:
    answer = row["target_answer"] if side == "target" else row["contrast_answer"]
    country = row["country"] if side == "target" else row["contrast_country"]
    title = row.get(f"{side}_evidence_title", "")
    url = row.get(f"{side}_evidence_url", "")
    text = " ".join([title, url, fetched.get("final_url", ""), fetched.get("text", "")])
    text_norm = normalize(text)
    ans_present = contains_any(text_norm, answer_aliases(answer))
    markers = COUNTRY_MARKERS.get(country, [country])
    country_present = contains_any(text_norm, markers)
    url_status = fetched.get("status", "")
    http_status = str(fetched.get("http_status", ""))
    hard_fail = []
    warnings = []
    if url_status == "missing_url":
        hard_fail.append(f"{side}_url_{url_status}")
    elif url_status == "http_error" and http_status == "429":
        warnings.append(f"{side}_url_rate_limited_429")
    elif url_status == "fetch_error":
        warnings.append(f"{side}_url_fetch_error")
    elif url_status != "ok":
        hard_fail.append(f"{side}_url_{url_status}_{http_status}".rstrip("_"))
    if url_status == "ok" and not ans_present:
        hard_fail.append(f"{side}_answer_absent_from_evidence_page")
    if url_status == "ok" and not country_present:
        warnings.append(f"{side}_country_marker_absent_from_evidence_page")
    if domain_authority(url) == "ordinary_web":
        warnings.append(f"{side}_ordinary_web_domain")
    return {
        f"{side}_url_status": url_status,
        f"{side}_http_status": fetched.get("http_status", ""),
        f"{side}_final_url": fetched.get("final_url", ""),
        f"{side}_answer_present": ans_present,
        f"{side}_country_marker_present": country_present,
        f"{side}_domain_type": domain_authority(url),
        f"{side}_fetch_error": fetched.get("error", ""),
        f"{side}_hard_failures": hard_fail,
        f"{side}_warnings": warnings,
    }


def question_leakage(row: dict) -> list[str]:
    q = normalize(row.get("question", ""))
    flags = []
    for label, country in [("target", row.get("country", "")), ("contrast", row.get("contrast_country", ""))]:
        for marker in COUNTRY_MARKERS.get(country, [country]):
            marker_norm = normalize(marker)
            if marker_norm and f" {marker_norm} " in f" {q} ":
                flags.append(f"{label}_country_marker_in_question")
                break
    return flags


def audit_row(row: dict, fetched_by_url: dict[str, dict]) -> dict:
    options = option_parts(row.get("options", []))
    hard_failures = []
    warnings = []
    if len(options) != 4:
        hard_failures.append("bad_option_count")
    if len({normalize(opt) for opt in options}) != len(options):
        hard_failures.append("duplicate_options")
    if sum(1 for opt in options if normalize(opt) == normalize(row.get("target_answer", ""))) != 1:
        hard_failures.append("target_answer_not_once_in_options")
    if sum(1 for opt in options if normalize(opt) == normalize(row.get("contrast_answer", ""))) < 1:
        hard_failures.append("contrast_answer_not_in_options")
    leakage = question_leakage(row)
    if leakage:
        hard_failures.extend(leakage)

    target = side_audit(row, "target", fetched_by_url.get(row.get("target_evidence_url", ""), {}))
    contrast = side_audit(row, "contrast", fetched_by_url.get(row.get("contrast_evidence_url", ""), {}))
    hard_failures.extend(target["target_hard_failures"])
    hard_failures.extend(contrast["contrast_hard_failures"])
    warnings.extend(target["target_warnings"])
    warnings.extend(contrast["contrast_warnings"])

    verdict = "pass"
    if hard_failures:
        verdict = "fail"
    elif warnings:
        verdict = "warn"
    out = {
        "id": row.get("id", ""),
        "source_row_id": row.get("source_row_id", ""),
        "country": row.get("country", ""),
        "contrast_country": row.get("contrast_country", ""),
        "topic": row.get("topic", ""),
        "year": row.get("year", ""),
        "question": row.get("question", ""),
        "target_answer": row.get("target_answer", ""),
        "contrast_answer": row.get("contrast_answer", ""),
        "target_evidence_url": row.get("target_evidence_url", ""),
        "contrast_evidence_url": row.get("contrast_evidence_url", ""),
        "weak_locale_risk": row.get("weak_locale_risk", ""),
        "llm_accept_or_restored": row.get("llm_accept_or_restored", ""),
        "web_audit_verdict": verdict,
        "hard_failures": " | ".join(sorted(set(hard_failures))),
        "warnings": " | ".join(sorted(set(warnings))),
    }
    out.update({k: v for k, v in target.items() if not k.endswith("_hard_failures") and not k.endswith("_warnings")})
    out.update({k: v for k, v in contrast.items() if not k.endswith("_hard_failures") and not k.endswith("_warnings")})
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="No-API web audit of LocalNewsQA ambiguous evidence URLs.")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--wiki-batch-size", type=int, default=25)
    parser.add_argument("--wiki-sleep", type=float, default=0.2)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    urls = sorted(
        {
            row.get("target_evidence_url", "")
            for row in rows
            if row.get("target_evidence_url")
        }
        | {
            row.get("contrast_evidence_url", "")
            for row in rows
            if row.get("contrast_evidence_url")
        }
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    fetched_by_url = {}
    started = time.time()
    wiki_urls = [url for url in urls if is_wikipedia_url(url)]
    other_urls = [url for url in urls if not is_wikipedia_url(url)]
    fetched_by_url.update(
        fetch_wikipedia_batches(
            wiki_urls,
            args.timeout,
            max(1, args.wiki_batch_size),
            max(0.0, args.wiki_sleep),
        )
    )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_url = {pool.submit(fetch_url, url, args.timeout): url for url in other_urls}
        for idx, future in enumerate(as_completed(future_to_url), start=1):
            url = future_to_url[future]
            fetched_by_url[url] = future.result()
            if idx % 100 == 0:
                print(json.dumps({"other_urls_fetched": idx, "total_other_urls": len(other_urls)}), flush=True)

    audits = [audit_row(row, fetched_by_url) for row in rows]
    write_csv(args.outdir / "web_audit_rows.csv", audits)
    write_csv(args.outdir / "web_audit_failures.csv", [row for row in audits if row["web_audit_verdict"] == "fail"])
    write_csv(args.outdir / "web_audit_warnings.csv", [row for row in audits if row["web_audit_verdict"] == "warn"])

    fetch_rows = [
        {
            "url": url,
            "status": data.get("status", ""),
            "http_status": data.get("http_status", ""),
            "final_url": data.get("final_url", ""),
            "content_type": data.get("content_type", ""),
            "error": data.get("error", ""),
        }
        for url, data in sorted(fetched_by_url.items())
    ]
    write_csv(args.outdir / "web_audit_url_fetches.csv", fetch_rows)

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "rows": len(rows),
        "unique_urls": len(urls),
        "elapsed_seconds": round(time.time() - started, 2),
        "verdict_counts": dict(Counter(row["web_audit_verdict"] for row in audits)),
        "hard_failure_counts": dict(
            Counter(
                part
                for row in audits
                for part in row["hard_failures"].split(" | ")
                if part
            ).most_common()
        ),
        "warning_counts": dict(
            Counter(
                part
                for row in audits
                for part in row["warnings"].split(" | ")
                if part
            ).most_common()
        ),
        "fetch_status_counts": dict(Counter(data.get("status", "") for data in fetched_by_url.values())),
        "paths": {
            "rows": str(args.outdir / "web_audit_rows.csv"),
            "failures": str(args.outdir / "web_audit_failures.csv"),
            "warnings": str(args.outdir / "web_audit_warnings.csv"),
            "url_fetches": str(args.outdir / "web_audit_url_fetches.csv"),
            "summary": str(args.outdir / "web_audit_summary.json"),
        },
    }
    (args.outdir / "web_audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
