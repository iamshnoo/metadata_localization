#!/usr/bin/env python3

import argparse
import importlib.util
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STRICT_DIR = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_semantic_gold_ambiguous_1700"
)
AUDIT_DIR = STRICT_DIR / "explicit_max_audit"
DEFAULT_INPUT = AUDIT_DIR / "localnewsqa_targetqa_explicit_style_max_paper_clean.jsonl"
DEFAULT_AMBIGUOUS = STRICT_DIR / "localnewsqa_ambiguous_semantic_gold_1700.jsonl"
DEFAULT_CACHE = AUDIT_DIR / "explicit_target_evidence_fetches.jsonl"
AUDIT_EXPLICIT_SCRIPT = ROOT / "qa_data/localnewsqa_core/48_audit_explicit_max_split.py"
AUDIT_FETCH_SCRIPT = ROOT / "qa_data/localnewsqa_core/32_web_audit_ambiguous_verifiable.py"

SOURCE_PRIORITY = {
    "explicit": 0,
    "ambiguous_salvaged_target": 1,
    "ambiguous_pool_salvaged_target_web_supported": 2,
}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def write_cache(path: Path, fetches: dict[str, dict[str, Any]]) -> None:
    write_jsonl(path, [fetches[url] for url in sorted(fetches)])


def norm(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def row_rank(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        SOURCE_PRIORITY.get(row.get("source_split_type", ""), 9),
        norm(row.get("target_answer", "")),
        norm(row.get("question", "")),
        row.get("source_row_id", ""),
    )


def strict_pass(row: dict[str, Any], fetches: dict[str, dict[str, Any]], ambiguous_ids: set[str], audit_mod: Any, fetch_mod: Any) -> bool:
    failures, warnings = audit_mod.audit_explicit_row(row, fetches, ambiguous_ids, fetch_mod)
    return not failures and not warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct-fetch explicit evidence until each country has a strict quota.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--ambiguous", type=Path, default=DEFAULT_AMBIGUOUS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--quota", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()

    audit_mod = load_module(AUDIT_EXPLICIT_SCRIPT, "audit48")
    fetch_mod = load_module(AUDIT_FETCH_SCRIPT, "fetch32")
    rows = read_jsonl(args.input)
    ambiguous_ids = {row.get("source_row_id", "") for row in read_jsonl(args.ambiguous)}
    fetches = load_fetch_cache(args.cache)
    by_country: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_url_country: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_country[row.get("country", "")].append(row)
        by_url_country[(row.get("country", ""), row.get("target_evidence_url", ""))].append(row)

    strict_ids = {
        row.get("source_row_id", "")
        for row in rows
        if strict_pass(row, fetches, ambiguous_ids, audit_mod, fetch_mod)
    }
    fetched_count = 0
    fetch_status_counts: Counter[str] = Counter()

    for country in sorted(by_country):
        country_rows = sorted(by_country[country], key=row_rank)
        country_strict = {row.get("source_row_id", "") for row in country_rows if row.get("source_row_id", "") in strict_ids}
        print(f"{country}: strict_start={len(country_strict)} quota={args.quota}", flush=True)
        if len(country_strict) >= args.quota:
            continue

        candidates = [row for row in country_rows if row.get("source_row_id", "") not in country_strict]
        seen_urls = set()
        for row in candidates:
            if len(country_strict) >= args.quota:
                break
            url = row.get("target_evidence_url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            cached = fetches.get(url, {})
            if not cached.get("ok"):
                fetched = fetch_mod.fetch_url(url, args.timeout, args.delay)
                fetches[url] = fetched
                fetched_count += 1
                fetch_status_counts[str(fetched.get("status_code", ""))] += 1
                if fetched_count % args.checkpoint_every == 0:
                    write_cache(args.cache, fetches)
                    print(
                        f"checkpoint fetched={fetched_count} status={dict(fetch_status_counts)} "
                        f"{country}_strict={len(country_strict)}",
                        flush=True,
                    )

            for related in by_url_country[(country, url)]:
                source_id = related.get("source_row_id", "")
                if source_id not in country_strict and strict_pass(related, fetches, ambiguous_ids, audit_mod, fetch_mod):
                    country_strict.add(source_id)
                    strict_ids.add(source_id)

        print(f"{country}: strict_end={len(country_strict)}", flush=True)
        write_cache(args.cache, fetches)
        time.sleep(2.0)

    write_cache(args.cache, fetches)
    strict_counts = Counter(row.get("country", "") for row in rows if row.get("source_row_id", "") in strict_ids)
    summary = {
        "quota": args.quota,
        "fetched": fetched_count,
        "fetch_status_counts": dict(fetch_status_counts),
        "strict_counts": dict(sorted(strict_counts.items())),
        "countries_below_quota": {
            country: count for country, count in sorted(strict_counts.items()) if count < args.quota
        },
        "cache": str(args.cache),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if any(count < args.quota for count in strict_counts.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
