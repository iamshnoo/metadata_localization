#!/usr/bin/env python3

import argparse
import csv
import html
import json
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import requests


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "what",
    "when",
    "which",
    "who",
    "with",
}


def normalize(text: str) -> str:
    text = html.unescape(str(text or ""))
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def keyword_tokens(text: str) -> List[str]:
    return [t for t in normalize(text).split() if len(t) >= 4 and t not in STOPWORDS]


def build_queries(row: Dict[str, str], side: str) -> List[str]:
    answer = row[f"{side}_answer"]
    country = row[f"{side}_country"]
    hint = row.get("evidence_hint", "")
    question = row.get("question", "").rstrip("?")
    question_prefix = " ".join(question.split()[:10])
    existing = row.get(f"{side}_query", "")
    queries: List[str] = []

    def add(q: str):
        q = re.sub(r"\s+", " ", str(q or "").strip())
        if q and q not in queries:
            queries.append(q)

    add(existing)
    add(f"\"{country}\" \"{answer}\"")
    add(f"{country} {answer}")
    if hint:
        add(f"{country} {answer} {hint}")
    if question_prefix:
        add(f"{country} {answer} {question_prefix}")
        add(f"{country} {question_prefix}")
    return queries[:4]


def bing_rss_search(session: requests.Session, query: str, max_results: int = 5) -> List[Dict[str, str]]:
    try:
        resp = session.get(
            "https://www.bing.com/search",
            params={"q": query, "format": "rss"},
            timeout=5,
            headers={"User-Agent": USER_AGENT},
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
    except Exception:
        return []

    hits = []
    for item in root.findall(".//item")[:max_results]:
        hits.append(
            {
                "url": item.findtext("link", default=""),
                "title": item.findtext("title", default=""),
                "snippet": item.findtext("description", default=""),
            }
        )
    return hits


def choose_hit(row: Dict[str, str], side: str, session: requests.Session) -> Dict[str, str]:
    answer = row[f"{side}_answer"]
    answer_norm = normalize(answer)
    answer_toks = set(keyword_tokens(answer))
    best = {
        "query": row.get(f"{side}_query", ""),
        "url": "",
        "title": "",
        "snippet": "",
        "excerpt": "",
        "match_type": "no_result",
    }
    best_score = -1

    for query in build_queries(row, side):
        hits = bing_rss_search(session, query, max_results=5)
        for rank, hit in enumerate(hits, start=1):
            title_norm = normalize(hit["title"])
            snippet_norm = normalize(hit["snippet"])
            combined = f"{title_norm} {snippet_norm}".strip()
            score = 0
            if answer_norm and answer_norm in combined:
                score += 6
            overlap = sum(tok in combined for tok in answer_toks)
            score += min(3, overlap)
            if normalize(row[f"{side}_country"]) in combined:
                score += 2
            if rank == 1:
                score += 1
            if score > best_score:
                best_score = score
                best = {
                    "query": query,
                    "url": hit["url"],
                    "title": hit["title"],
                    "snippet": hit["snippet"],
                    "excerpt": hit["snippet"],
                    "match_type": "bing_rss_match" if score >= 6 else "bing_rss_candidate",
                }
            if score >= 7:
                return best
        if best.get("url"):
            return best
    return best


def fill_row(args: Tuple[int, Dict[str, str]]) -> Tuple[int, Dict[str, str]]:
    idx, row = args
    row = dict(row)
    session = requests.Session()
    for side in ("target", "contrast"):
        if not str(row.get(f"{side}_answer", "")).strip():
            continue
        if row.get(f"{side}_evidence_url", "").strip():
            continue
        ev = choose_hit(row, side, session)
        row[f"{side}_query"] = ev["query"]
        row[f"{side}_evidence_url"] = ev["url"]
        row[f"{side}_evidence_title"] = ev["title"]
        row[f"{side}_evidence_snippet"] = ev["snippet"]
        row[f"{side}_evidence_excerpt"] = ev["excerpt"]
        row[f"{side}_match_type"] = ev["match_type"]
    return idx, row


def main():
    parser = argparse.ArgumentParser(description="Fast-fill blank evidence URLs for LocalNewsQA human validation.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    with open(args.input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    before_target = sum(bool((r.get("target_evidence_url") or "").strip()) for r in rows)
    before_contrast = sum(bool((r.get("contrast_evidence_url") or "").strip()) for r in rows)

    indexed = list(enumerate(rows))
    outputs: List[Dict[str, str]] = [None] * len(indexed)  # type: ignore
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for idx, row in ex.map(fill_row, indexed):
            outputs[idx] = row

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(outputs)

    summary = {
        "rows": len(outputs),
        "before_target_urls": before_target,
        "before_contrast_urls": before_contrast,
        "after_target_urls": sum(bool((r.get("target_evidence_url") or "").strip()) for r in outputs),
        "after_contrast_urls": sum(bool((r.get("contrast_evidence_url") or "").strip()) for r in outputs),
        "after_both_urls": sum(
            bool((r.get("target_evidence_url") or "").strip())
            and bool((r.get("contrast_evidence_url") or "").strip())
            for r in outputs
        ),
        "after_either_urls": sum(
            bool((r.get("target_evidence_url") or "").strip())
            or bool((r.get("contrast_evidence_url") or "").strip())
            for r in outputs
        ),
        "output_csv": str(out_path),
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
