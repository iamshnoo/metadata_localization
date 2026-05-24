#!/usr/bin/env python3

import argparse
import csv
import html
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import requests
from bs4 import BeautifulSoup
from datasets import load_dataset


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

COUNTRY_MARKERS = {
    "Bangladesh": ["bangladesh", "bangladeshi"],
    "Canada": ["canada", "canadian"],
    "Ghana": ["ghana", "ghanaian"],
    "Hong Kong": ["hong kong"],
    "India": ["india", "indian"],
    "Ireland": ["ireland", "irish"],
    "Jamaica": ["jamaica", "jamaican"],
    "Kenya": ["kenya", "kenyan"],
    "Malaysia": ["malaysia", "malaysian"],
    "Nigeria": ["nigeria", "nigerian"],
    "Pakistan": ["pakistan", "pakistani"],
    "Philippines": ["philippines", "philippine", "filipino"],
    "South Africa": ["south africa", "south african"],
    "Sri Lanka": ["sri lanka", "sri lankan"],
    "Tanzania": ["tanzania", "tanzanian"],
    "United Kingdom": ["united kingdom", "uk", "britain", "british"],
    "United States": ["united states", "usa", "u s", "u.s", "american"],
}


def normalize(text: str) -> str:
    text = html.unescape(str(text or ""))
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_excerpt(text: str, needle: str, window: int = 140) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    low_text = raw.lower()
    low_needle = str(needle or "").lower()
    idx = low_text.find(low_needle)
    if idx == -1:
        return raw[: 2 * window].strip()
    start = max(0, idx - window)
    end = min(len(raw), idx + len(needle) + window)
    return raw[start:end].strip()


def country_in_question(question: str, country: str) -> bool:
    q = normalize(question)
    c = normalize(country)
    return bool(c and c in q)


def build_queries(question: str, country: str, answer: str, evidence_hint: str) -> List[str]:
    hint = str(evidence_hint or "").strip()
    question_keywords = " ".join(str(question).strip().rstrip("?").split()[:8])
    queries = [f"\"{country}\" \"{answer}\"", f"\"{answer}\""]
    contextual = [f"\"{country}\"", f"\"{answer}\""]
    if hint:
        contextual.append(hint)
    if question_keywords:
        contextual.append(question_keywords)
    contextual_query = " ".join(contextual)
    if contextual_query not in queries:
        queries.append(contextual_query)
    answer_contextual = [f"\"{answer}\""]
    if hint:
        answer_contextual.append(hint)
    if question_keywords:
        answer_contextual.append(question_keywords)
    answer_contextual_query = " ".join(answer_contextual)
    if answer_contextual_query not in queries:
        queries.append(answer_contextual_query)
    return queries


def fetch_page_text(url: str, timeout: int = 12) -> str:
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
            allow_redirects=True,
        )
        if resp.status_code != 200 or "text/html" not in resp.headers.get("content-type", ""):
            return ""
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(" ", strip=True)
        return re.sub(r"\s+", " ", text)
    except Exception:
        return ""


def wikipedia_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "utf8": 1,
                "format": "json",
                "srlimit": max_results,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=20,
        )
        resp.raise_for_status()
    except Exception:
        return []

    data = resp.json().get("query", {}).get("search", [])
    hits = []
    for row in data[:max_results]:
        title = row.get("title", "")
        href = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
        hits.append(
            {
                "url": href,
                "title": title,
                "snippet": BeautifulSoup(row.get("snippet", ""), "lxml").get_text(" ", strip=True),
            }
        )
    return hits


def wikipedia_extract(title: str) -> str:
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "titles": title,
                "format": "json",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=20,
        )
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", "")
            if extract:
                return re.sub(r"\s+", " ", extract)
    except Exception:
        pass
    return ""


def search_evidence(
    queries: List[str],
    answer: str,
    country: str,
    max_results: int = 5,
) -> Dict[str, str]:
    answer_norm = normalize(answer)
    country_markers = COUNTRY_MARKERS.get(country, [normalize(country)])

    fallback = {
        "query": queries[0] if queries else "",
        "url": "",
        "title": "",
        "snippet": "",
        "excerpt": "",
        "match_type": "no_result",
        "error": "",
    }

    for query in queries:
        try:
            results = wikipedia_search(query, max_results=max_results)
        except Exception as e:
            fallback["error"] = f"{type(e).__name__}: {e}"
            continue
        for rank, row in enumerate(results, start=1):
            url = row.get("href") or row.get("url") or ""
            title = row.get("title") or ""
            snippet = row.get("body") or row.get("snippet") or ""
            combined = normalize(f"{title} {snippet}")
            record = {
                "query": query,
                "url": url,
                "title": title,
                "snippet": snippet,
                "excerpt": "",
                "match_type": "search_result",
                "error": "",
                "rank": str(rank),
            }
            if not fallback["url"]:
                fallback.update(record)

            title_norm = normalize(title)
            snippet_country_ok = any(marker in combined for marker in country_markers)
            answer_country_in_title = any(marker in title_norm for marker in country_markers)
            title_exact = answer_norm == title_norm
            single_token = len(str(answer).split()) == 1

            if (
                answer_norm
                and answer_norm in combined
                and (snippet_country_ok or answer_country_in_title or title_exact)
                and (not single_token or title_exact)
            ):
                record["match_type"] = "wikipedia_search_match"
                record["excerpt"] = extract_excerpt(f"{title}. {snippet}", answer)
                return record
            if title:
                page_text = wikipedia_extract(title)
                if page_text:
                    page_norm = normalize(page_text)
                    record["excerpt"] = extract_excerpt(page_text, answer)
                    page_country_ok = any(marker in page_norm for marker in country_markers)
                    if (
                        answer_norm
                        and answer_norm in page_norm
                        and (page_country_ok or answer_country_in_title or title_exact)
                        and (not single_token or title_exact)
                    ):
                        record["match_type"] = "wikipedia_extract_match"
                        return record
    return fallback


def sample_rows(dataset_name: str, split: str, per_country: int, seed: int) -> List[Dict]:
    ds = load_dataset(dataset_name, split=split)
    ambiguous = [dict(row) for row in ds if str(row.get("split_type", "")).lower() == "ambiguous"]
    by_country: Dict[str, List[Dict]] = defaultdict(list)
    for row in ambiguous:
        by_country[str(row["country"]).strip()].append(row)

    rng = random.Random(seed)
    sampled = []
    for country in sorted(by_country):
        rows = by_country[country]
        if len(rows) < per_country:
            raise SystemExit(
                f"Country '{country}' has only {len(rows)} ambiguous rows; cannot sample {per_country}."
            )
        rng.shuffle(rows)
        sampled.extend(rows[:per_country])
    return sampled


def row_to_output(row: Dict, target_ev: Dict[str, str], contrast_ev: Dict[str, str]) -> Dict[str, str]:
    no_leakage = not country_in_question(row["question"], row["target_country"])
    target_ok = target_ev.get("match_type") in {"wikipedia_search_match", "wikipedia_extract_match"}
    contrast_ok = contrast_ev.get("match_type") in {"wikipedia_search_match", "wikipedia_extract_match"}
    locale_dep = target_ok and contrast_ok and row["target_answer"] != row["contrast_answer"]

    notes = (
        f"target={row['target_country']}:{row['target_answer']} [{target_ev.get('match_type','')}]; "
        f"contrast={row['contrast_country']}:{row['contrast_answer']} [{contrast_ev.get('match_type','')}]"
    )

    return {
        "id": row.get("generation_custom_id", ""),
        "country": row.get("country", ""),
        "continent": row.get("continent", ""),
        "topic": row.get("topic", ""),
        "year": row.get("year", ""),
        "question": row.get("question", ""),
        "options": " || ".join(f"{chr(65 + idx)}: {opt}" for idx, opt in enumerate(row.get("options", []))),
        "target_country": row.get("target_country", ""),
        "contrast_country": row.get("contrast_country", ""),
        "target_answer": row.get("target_answer", ""),
        "contrast_answer": row.get("contrast_answer", ""),
        "evidence_hint": row.get("evidence_hint", ""),
        "target_query": target_ev.get("query", ""),
        "target_evidence_url": target_ev.get("url", ""),
        "target_evidence_title": target_ev.get("title", ""),
        "target_evidence_snippet": target_ev.get("snippet", ""),
        "target_evidence_excerpt": target_ev.get("excerpt", ""),
        "target_match_type": target_ev.get("match_type", ""),
        "contrast_query": contrast_ev.get("query", ""),
        "contrast_evidence_url": contrast_ev.get("url", ""),
        "contrast_evidence_title": contrast_ev.get("title", ""),
        "contrast_evidence_snippet": contrast_ev.get("snippet", ""),
        "contrast_evidence_excerpt": contrast_ev.get("excerpt", ""),
        "contrast_match_type": contrast_ev.get("match_type", ""),
        "judge_target_factuality": "yes" if target_ok else "",
        "judge_locale_dependence": "yes" if locale_dep else "",
        "judge_no_explicit_leakage": "yes" if no_leakage else "no",
        "annotator_notes": notes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build a web-backed human-validation sheet for LocalNewsQA ambiguous items."
    )
    parser.add_argument("--dataset", default="iamshnoo/LocalNewsQA")
    parser.add_argument("--split", default="train")
    parser.add_argument("--per-country", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--progress-jsonl", default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for smoke testing")
    args = parser.parse_args()

    rows = sample_rows(args.dataset, args.split, args.per_country, args.seed)
    if args.limit is not None:
        rows = rows[: args.limit]

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_f = None
    if args.progress_jsonl:
        Path(args.progress_jsonl).parent.mkdir(parents=True, exist_ok=True)
        progress_f = Path(args.progress_jsonl).open("w", encoding="utf-8")

    outputs: List[Dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        target_queries = build_queries(
            row["question"], row["target_country"], row["target_answer"], row.get("evidence_hint", "")
        )
        contrast_queries = build_queries(
            row["question"], row["contrast_country"], row["contrast_answer"], row.get("evidence_hint", "")
        )
        target_ev = search_evidence(
            target_queries, row["target_answer"], row["target_country"], max_results=args.max_results
        )
        time.sleep(args.sleep_seconds)
        contrast_ev = search_evidence(
            contrast_queries, row["contrast_answer"], row["contrast_country"], max_results=args.max_results
        )
        time.sleep(args.sleep_seconds)

        out_row = row_to_output(row, target_ev, contrast_ev)
        outputs.append(out_row)
        if progress_f is not None:
            progress_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            progress_f.flush()
        if idx % 25 == 0:
            print(f"Processed {idx}/{len(rows)}")

    if progress_f is not None:
        progress_f.close()

    fieldnames = list(outputs[0].keys()) if outputs else []
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(outputs)

    hit_types = {"wikipedia_search_match", "wikipedia_extract_match"}
    target_hits = sum(1 for r in outputs if r["target_match_type"] in hit_types)
    contrast_hits = sum(1 for r in outputs if r["contrast_match_type"] in hit_types)
    locale_hits = sum(1 for r in outputs if r["judge_locale_dependence"] == "yes")
    summary = {
        "total_rows": len(outputs),
        "target_evidence_hits": target_hits,
        "contrast_evidence_hits": contrast_hits,
        "locale_dependence_prefilled_yes": locale_hits,
        "output_csv": str(output_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
