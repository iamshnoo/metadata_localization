#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[2]
STRICT_DIR = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_semantic_gold_ambiguous_1700"
)
DEFAULT_INPUT = STRICT_DIR / "localnewsqa_targetqa_explicit_style_max_target_supported.jsonl"
DEFAULT_AMBIGUOUS = STRICT_DIR / "localnewsqa_ambiguous_semantic_gold_1700.jsonl"
DEFAULT_OUTDIR = STRICT_DIR / "explicit_max_audit"
AUDIT_SCRIPT = ROOT / "qa_data/localnewsqa_core/32_web_audit_ambiguous_verifiable.py"
SEED_FETCHES = [
    ROOT / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_audit_ambiguous_1700/url_fetches.jsonl",
    STRICT_DIR / "semantic_gold_selected_evidence_fetches.jsonl",
    DEFAULT_OUTDIR / "explicit_target_evidence_fetches.jsonl",
]

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

GENERIC_EVIDENCE_TITLES = {
    "airport",
    "attorney general",
    "cabinet",
    "capital city",
    "central bank",
    "christmas day",
    "city",
    "court",
    "cricket",
    "currency",
    "department of justice",
    "flag",
    "football",
    "good friday",
    "governor general",
    "governor-general",
    "high court",
    "house of representatives",
    "independence day",
    "labour day",
    "minister",
    "ministry of education",
    "ministry of health",
    "national assembly",
    "netball",
    "new year",
    "newspaper",
    "ombudsman",
    "parliament",
    "president",
    "prime minister",
    "radio",
    "rugby",
    "school",
    "senate",
    "speaker",
    "state house",
    "stock exchange",
    "supreme court",
    "television",
    "university",
}
REVIEWER_VISIBLE_WARNINGS = {
    "duplicate_question_text",
    "target_country_marker_not_found_in_evidence",
    "target_evidence_text_short",
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


def norm(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_phrase(text: str, phrase: str) -> bool:
    phrase_norm = norm(phrase)
    return bool(phrase_norm) and f" {phrase_norm} " in f" {norm(text)} "


def option_count(options: list[str], answer: str) -> int:
    answer_norm = norm(answer)
    return sum(1 for option in options if norm(option) == answer_norm)


def country_marker_in_question(row: dict) -> bool:
    question = row.get("question", "")
    return any(contains_phrase(question, marker) for marker in COUNTRY_MARKERS.get(row.get("country", ""), []))


def question_quality_issues(question: str) -> list[str]:
    checks = {
        "placeholder_relevant": r"\brelevant\b",
        "placeholder_matches_local": r"\bmatches local\b",
        "placeholder_fits_context": r"\bfits .*context\b",
        "fallback_which_answer": r"^which (?:answer|option)\b",
        "trailing_fragment": r"\b(?:is|are|was|were|the|of|in|at|by)\??$",
        "double_period": r"\.\?",
    }
    found = []
    if not str(question or "").strip().endswith("?"):
        found.append("not_question_mark")
    for name, pattern in checks.items():
        if re.search(pattern, str(question or ""), flags=re.IGNORECASE):
            found.append(name)
    return found


def generic_evidence_key(row: dict) -> str:
    title = norm(row.get("target_evidence_title", ""))
    url_tail = norm(str(row.get("target_evidence_url", "")).split("/")[-1].replace("_", " "))
    if title in GENERIC_EVIDENCE_TITLES:
        return title
    if url_tail in GENERIC_EVIDENCE_TITLES:
        return url_tail
    return ""


def is_reviewer_visible_warning(warning: str) -> bool:
    return warning in REVIEWER_VISIBLE_WARNINGS or warning.startswith("generic_target_evidence:")


def load_fetch_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("url"):
                    out[row["url"]] = row
    return out


def write_fetch_cache(path: Path, fetches: dict[str, dict]) -> None:
    rows = [fetches[url] for url in sorted(fetches)]
    write_jsonl(path, rows)


def fetch_missing(
    urls: list[str],
    fetches: dict[str, dict],
    audit_mod: Any,
    workers: int,
    timeout: float,
    delay: float,
    retry_failed: bool,
) -> None:
    missing = [
        url
        for url in urls
        if url and (url not in fetches or (retry_failed and not fetches.get(url, {}).get("ok")))
    ]
    if not missing:
        return
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_url = {pool.submit(audit_mod.fetch_url, url, timeout, delay): url for url in missing}
        for future in as_completed(future_to_url):
            row = future.result()
            fetches[row["url"]] = row
            done += 1
            if done % 500 == 0:
                print(f"fetched {done}/{len(missing)} missing target evidence pages", flush=True)


def wikipedia_title_from_url(url: str) -> str:
    try:
        parsed = urlparse(url)
    except Exception:
        return ""
    if parsed.netloc not in {"en.wikipedia.org", "en.m.wikipedia.org"}:
        return ""
    prefix = "/wiki/"
    if not parsed.path.startswith(prefix):
        return ""
    return unquote(parsed.path[len(prefix) :]).replace("_", " ").strip()


def hydrate_wikipedia_api(urls: list[str], fetches: dict[str, dict], batch_size: int, sleep: float) -> None:
    try:
        import requests
    except ImportError:
        return
    api_url = "https://en.wikipedia.org/w/api.php"
    pending = [(url, wikipedia_title_from_url(url)) for url in urls if wikipedia_title_from_url(url)]
    pending = [(url, title) for url, title in pending if not fetches.get(url, {}).get("ok")]
    if not pending:
        return
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        titles = [title for _, title in batch]
        try:
            response = requests.get(
                api_url,
                params={
                    "action": "query",
                    "format": "json",
                    "prop": "extracts|info",
                    "explaintext": 1,
                    "exintro": 0,
                    "redirects": 1,
                    "inprop": "url",
                    "titles": "|".join(titles),
                },
                timeout=30,
                headers={"User-Agent": "LocalNewsQA-no-api-explicit-audit/1.0"},
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            error = str(exc)[:500]
            for url, _ in batch:
                old = fetches.get(url, {})
                if not old.get("ok"):
                    fetches[url] = {
                        "url": url,
                        "ok": False,
                        "status_code": "",
                        "final_url": "",
                        "content_type": "application/json",
                        "title": "",
                        "text": "",
                        "text_len": 0,
                        "error": f"wikipedia_api_failed:{error}",
                        "elapsed_sec": 0.0,
                    }
            continue

        redirects = {item.get("from", ""): item.get("to", "") for item in data.get("query", {}).get("redirects", [])}
        normalized = {item.get("from", ""): item.get("to", "") for item in data.get("query", {}).get("normalized", [])}
        pages_by_title = {page.get("title", ""): page for page in data.get("query", {}).get("pages", {}).values()}
        for url, original_title in batch:
            title = redirects.get(original_title, normalized.get(original_title, original_title))
            title = redirects.get(title, normalized.get(title, title))
            page = pages_by_title.get(title) or pages_by_title.get(original_title)
            if not page or "missing" in page:
                old = fetches.get(url, {})
                if not old.get("ok"):
                    fetches[url] = {
                        "url": url,
                        "ok": False,
                        "status_code": "404",
                        "final_url": "",
                        "content_type": "application/json",
                        "title": original_title,
                        "text": "",
                        "text_len": 0,
                        "error": "wikipedia_api_page_missing",
                        "elapsed_sec": 0.0,
                    }
                continue
            text = re.sub(r"\s+", " ", str(page.get("extract", "") or "")).strip()
            fetches[url] = {
                "url": url,
                "ok": bool(text),
                "status_code": "200",
                "final_url": page.get("fullurl", url),
                "content_type": "application/json",
                "title": page.get("title", original_title),
                "text": text[:120_000],
                "text_len": len(text[:120_000]),
                "error": "" if text else "wikipedia_api_empty_extract",
                "elapsed_sec": 0.0,
            }
        print(f"wikipedia api hydrated {min(start + batch_size, len(pending))}/{len(pending)} failed/missing pages", flush=True)
        if sleep > 0:
            time.sleep(sleep)


def evidence_blob(row: dict, fetch: dict) -> str:
    return " ".join(
        [
            row.get("target_evidence_url", ""),
            row.get("target_evidence_title", ""),
            row.get("target_evidence_excerpt", ""),
            fetch.get("title", ""),
            fetch.get("text", ""),
        ]
    )


def answer_found(row: dict, fetch: dict, audit_mod: Any) -> bool:
    blob_norm = audit_mod.normalize_text(evidence_blob(row, fetch))
    return audit_mod.contains_any(blob_norm, audit_mod.answer_aliases(row.get("target_answer", "")))


def answer_found_in_metadata(row: dict, audit_mod: Any) -> bool:
    blob = " ".join(
        [
            row.get("target_evidence_url", ""),
            row.get("target_evidence_title", ""),
            row.get("target_evidence_excerpt", ""),
        ]
    )
    blob_norm = audit_mod.normalize_text(blob)
    return audit_mod.contains_any(blob_norm, audit_mod.answer_aliases(row.get("target_answer", "")))


def country_found_in_evidence(row: dict, fetch: dict, audit_mod: Any) -> bool:
    blob_norm = audit_mod.normalize_text(evidence_blob(row, fetch))
    return audit_mod.contains_any(blob_norm, audit_mod.marker_aliases(row.get("country", "")))


def audit_explicit_row(row: dict, fetches: dict[str, dict], ambiguous_ids: set[str], audit_mod: Any) -> tuple[list[str], list[str]]:
    failures = []
    warnings = []
    options = row.get("options") or []
    norm_options = [norm(option) for option in options]
    if row.get("source_row_id", "") in ambiguous_ids:
        failures.append("overlaps_ambiguous_source_id")
    if row.get("split_type") != "explicit":
        failures.append("split_type_not_explicit")
    if row.get("ambiguity_flag") is not False:
        failures.append("ambiguity_flag_not_false")
    if len(options) != 4:
        failures.append("option_count_not_4")
    if len(norm_options) != len(set(norm_options)):
        failures.append("duplicate_normalized_options")
    if option_count(options, row.get("target_answer", "")) != 1:
        failures.append("target_answer_not_exactly_once")
    if norm(row.get("target_answer", "")) != norm(row.get("correct_answer", "")):
        failures.append("correct_answer_not_target_answer")
    if not country_marker_in_question(row):
        failures.append("target_country_marker_missing_in_question")
    failures.extend(f"question:{issue}" for issue in question_quality_issues(row.get("question", "")))

    url = row.get("target_evidence_url", "")
    fetch = fetches.get(url, {})
    metadata_answer_found = answer_found_in_metadata(row, audit_mod)
    if not url:
        failures.append("target_evidence_url_missing")
    elif not fetch:
        if metadata_answer_found:
            warnings.append("target_evidence_fetch_missing_but_metadata_supports_answer")
        else:
            failures.append("target_evidence_fetch_missing")
    elif not fetch.get("ok"):
        status = str(fetch.get("status_code", ""))
        if metadata_answer_found and status not in {"403", "404"}:
            status_label = {
                "": "failed",
                "200": "empty",
                "429": "rate_limited",
            }.get(status, f"status_{status}")
            warnings.append(f"target_evidence_fetch_{status_label}_but_metadata_supports_answer")
        else:
            failures.append("target_evidence_fetch_failed")
    else:
        if not answer_found(row, fetch, audit_mod):
            failures.append("target_answer_not_found_in_evidence")
        if not country_found_in_evidence(row, fetch, audit_mod):
            warnings.append("target_country_marker_not_found_in_evidence")
        if int(fetch.get("text_len", 0) or 0) < 500:
            warnings.append("target_evidence_text_short")

    generic_key = generic_evidence_key(row)
    if generic_key:
        warnings.append(f"generic_target_evidence:{generic_key}")
    return failures, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the maximum explicit LocalNewsQA split.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--ambiguous", type=Path, default=DEFAULT_AMBIGUOUS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--fetch-missing", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--hydrate-wikipedia-api", action="store_true")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--wiki-batch-size", type=int, default=50)
    parser.add_argument("--wiki-sleep", type=float, default=0.05)
    parser.add_argument("--exclude-warning-rows", action="store_true")
    args = parser.parse_args()

    audit_mod = load_module(AUDIT_SCRIPT, "audit32")
    rows = read_jsonl(args.input)
    ambiguous_ids = {row["source_row_id"] for row in read_jsonl(args.ambiguous)}

    args.outdir.mkdir(parents=True, exist_ok=True)
    fetch_path = args.outdir / "explicit_target_evidence_fetches.jsonl"
    fetches = {}
    for seed in SEED_FETCHES:
        fetches.update(load_fetch_cache(seed))
    fetches.update(load_fetch_cache(fetch_path))
    target_urls = sorted({row.get("target_evidence_url", "") for row in rows if row.get("target_evidence_url", "")})
    if args.fetch_missing:
        fetch_missing(
            target_urls,
            fetches,
            audit_mod,
            workers=args.workers,
            timeout=args.timeout,
            delay=args.delay,
            retry_failed=args.retry_failed,
        )
        write_fetch_cache(fetch_path, {url: fetches[url] for url in target_urls if url in fetches})
    if args.hydrate_wikipedia_api:
        hydrate_wikipedia_api(target_urls, fetches, batch_size=args.wiki_batch_size, sleep=args.wiki_sleep)
        write_fetch_cache(fetch_path, {url: fetches[url] for url in target_urls if url in fetches})

    failures = []
    warnings = []
    clean_rows = []
    paper_clean_rows = []
    clean_strict_rows = []
    source_ids = [row.get("source_row_id", "") for row in rows]
    duplicate_source_ids = {source_id for source_id, count in Counter(source_ids).items() if count > 1}
    exact_questions = Counter(row.get("question", "") for row in rows)
    duplicate_questions = {question for question, count in exact_questions.items() if count > 1}

    for row in rows:
        row_failures, row_warnings = audit_explicit_row(row, fetches, ambiguous_ids, audit_mod)
        if row.get("source_row_id", "") in duplicate_source_ids:
            row_failures.append("duplicate_source_id")
        if row.get("question", "") in duplicate_questions:
            row_warnings.append("duplicate_question_text")
        if row_failures:
            failures.append(
                {
                    "source_row_id": row.get("source_row_id", ""),
                    "country": row.get("country", ""),
                    "source_split_type": row.get("source_split_type", ""),
                    "question": row.get("question", ""),
                    "target_answer": row.get("target_answer", ""),
                    "target_evidence_title": row.get("target_evidence_title", ""),
                    "target_evidence_url": row.get("target_evidence_url", ""),
                    "failures": " | ".join(sorted(set(row_failures))),
                    "warnings": " | ".join(sorted(set(row_warnings))),
                }
            )
        if row_warnings:
            warnings.append(
                {
                    "source_row_id": row.get("source_row_id", ""),
                    "country": row.get("country", ""),
                    "source_split_type": row.get("source_split_type", ""),
                    "question": row.get("question", ""),
                    "target_answer": row.get("target_answer", ""),
                    "target_evidence_title": row.get("target_evidence_title", ""),
                    "target_evidence_url": row.get("target_evidence_url", ""),
                    "warnings": " | ".join(sorted(set(row_warnings))),
                }
            )
        if not row_failures:
            clean_rows.append(row)
            if not any(is_reviewer_visible_warning(warning) for warning in row_warnings):
                paper_clean_rows.append(row)
            if not row_warnings:
                clean_strict_rows.append(row)

    failures_path = args.outdir / "explicit_max_quality_failures.csv"
    warnings_path = args.outdir / "explicit_max_quality_warnings.csv"
    clean_path = args.outdir / "localnewsqa_targetqa_explicit_style_max_clean.jsonl"
    clean_csv_path = args.outdir / "localnewsqa_targetqa_explicit_style_max_clean.csv"
    paper_clean_path = args.outdir / "localnewsqa_targetqa_explicit_style_max_paper_clean.jsonl"
    paper_clean_csv_path = args.outdir / "localnewsqa_targetqa_explicit_style_max_paper_clean.csv"
    strict_clean_path = args.outdir / "localnewsqa_targetqa_explicit_style_max_strict_no_warnings.jsonl"
    summary_path = args.outdir / "explicit_max_quality_summary.json"
    write_csv(failures_path, failures)
    write_csv(warnings_path, warnings)
    write_jsonl(clean_path, clean_rows)
    write_csv(clean_csv_path, clean_rows)
    write_jsonl(paper_clean_path, paper_clean_rows)
    write_csv(paper_clean_csv_path, paper_clean_rows)
    write_jsonl(strict_clean_path, clean_strict_rows)

    failure_counts = Counter(
        issue
        for row in failures
        for issue in row["failures"].split(" | ")
        if issue
    )
    warning_counts = Counter(
        issue
        for row in warnings
        for issue in row["warnings"].split(" | ")
        if issue
    )
    summary = {
        "input": str(args.input),
        "rows": len(rows),
        "target_unique_urls": len(target_urls),
        "target_fetch_coverage": sum(1 for url in target_urls if url in fetches),
        "target_fetch_ok": sum(1 for url in target_urls if fetches.get(url, {}).get("ok")),
        "failure_rows": len(failures),
        "warning_rows": len(warnings),
        "clean_rows_no_failures": len(clean_rows),
        "paper_clean_rows_no_failures_no_reviewer_visible_warnings": len(paper_clean_rows),
        "strict_clean_rows_no_failures_or_warnings": len(clean_strict_rows),
        "failure_counts": dict(failure_counts),
        "warning_counts": dict(warning_counts),
        "source_split_type_counts": dict(Counter(row.get("source_split_type", "") for row in rows)),
        "clean_source_split_type_counts": dict(Counter(row.get("source_split_type", "") for row in clean_rows)),
        "paper_clean_source_split_type_counts": dict(
            Counter(row.get("source_split_type", "") for row in paper_clean_rows)
        ),
        "paper_clean_policy": (
            "keeps rows with no hard failures and excludes reviewer-visible warnings: "
            "generic target evidence, duplicate exact question text, short fetched evidence, "
            "or fetched evidence that lacks a target-country marker; cache/fetch warnings are allowed "
            "when target evidence metadata supports the answer"
        ),
        "paths": {
            "failures": str(failures_path),
            "warnings": str(warnings_path),
            "clean_jsonl": str(clean_path),
            "clean_csv": str(clean_csv_path),
            "paper_clean_jsonl": str(paper_clean_path),
            "paper_clean_csv": str(paper_clean_csv_path),
            "strict_clean_jsonl": str(strict_clean_path),
            "fetches": str(fetch_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
