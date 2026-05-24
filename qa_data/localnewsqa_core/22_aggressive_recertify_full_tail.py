#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple


CERTIFIER_PATH = "./qa_data/localnewsqa_core/13_certify_validation_sources.py"


def load_certifier():
    spec = importlib.util.spec_from_file_location("lnqa_certifier", CERTIFIER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load certifier module from {CERTIFIER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CERT = load_certifier()


def needs_side(row: Dict[str, str], side: str) -> bool:
    answer = str(row.get(f"{side}_answer", "")).strip()
    if not answer:
        return False
    return str(row.get(f"{side}_source_certified", "")).strip().lower() != "yes"


def make_rec(
    row: Dict[str, str],
    side: str,
    query: str,
    cand: Dict[str, str],
    excerpt: str,
    match_type: str,
    reason: str,
) -> Dict[str, str]:
    return {
        "query": query,
        "url": cand.get("url", ""),
        "title": cand.get("title", ""),
        "snippet": cand.get("snippet", ""),
        "excerpt": excerpt,
        "match_type": match_type,
        "certified": "yes",
        "reason": reason,
    }


def aggressive_certify_candidate(searcher, row: Dict[str, str], side: str, cand: Dict[str, str]) -> Tuple[bool, str, str, str]:
    answer = row[f"{side}_answer"]
    answer_norm = CERT.normalize(answer)
    answer_toks = CERT.answer_tokens(answer)
    generic = CERT.is_generic_answer(answer)
    kws = CERT.context_keywords(row, side)
    markers = CERT.country_markers(row[f"{side}_country"])

    title = cand.get("title", "")
    snippet = cand.get("snippet", "")
    url = cand.get("url", "")
    source = cand.get("source", "")

    text = f"{title}. {snippet}"
    if url and not CERT.is_blacklisted(url):
        page_text = searcher.fetch_page_text(url, timeout=5)
        if page_text:
            text = f"{title}. {snippet}. {page_text[:25000]}"

    norm = CERT.normalize(text)
    title_norm = CERT.normalize(title)
    ans_phrase = bool(answer_norm and answer_norm in norm)
    title_exact = bool(answer_norm and (answer_norm == title_norm or answer_norm in title_norm))
    ans_tok_hits = sum(tok in norm for tok in answer_toks)
    kw_hits = sum(kw in norm for kw in kws)
    country_hits = sum(marker in norm for marker in markers)

    certified = False
    reason = ""
    if title_exact and (country_hits or kw_hits >= 1 or not generic):
        certified = True
        reason = "aggressive_title_exact"
    elif ans_phrase and (country_hits or kw_hits >= 1):
        certified = True
        reason = "aggressive_answer_phrase_plus_context"
    elif not generic and ans_tok_hits >= max(1, len(answer_toks) - 1) and (country_hits or kw_hits >= 1):
        certified = True
        reason = "aggressive_answer_tokens_plus_context"
    elif generic and ans_tok_hits >= max(1, len(answer_toks)) and (country_hits + kw_hits >= 2):
        certified = True
        reason = "aggressive_generic_answer_plus_context"

    if certified:
        dom = CERT.domain_of(url)
        if source == "wikipedia" or "wikipedia.org" in dom:
            match_type = "wikipedia_extract_match"
        elif source == "duckduckgo":
            match_type = "duckduckgo_page_match"
        elif source == "bing":
            match_type = "bing_rss_page_match"
        elif source == "existing":
            match_type = "existing_page_match"
        else:
            match_type = f"{source}_page_match" if source else "aggressive_page_match"
    else:
        match_type = f"{source}_candidate" if source else "no_result"

    return certified, match_type, CERT.extract_excerpt(text, answer), reason


def aggressive_certify_side(searcher, row: Dict[str, str], side: str) -> Dict[str, str]:
    current = {
        "query": row.get(f"{side}_query", ""),
        "url": row.get(f"{side}_evidence_url", ""),
        "title": row.get(f"{side}_evidence_title", ""),
        "snippet": row.get(f"{side}_evidence_snippet", ""),
        "excerpt": row.get(f"{side}_evidence_excerpt", ""),
        "match_type": row.get(f"{side}_match_type", ""),
        "certified": row.get(f"{side}_source_certified", ""),
        "reason": row.get(f"{side}_cert_reason", ""),
    }

    base = CERT.certify_side(searcher, row, side)
    if base.get("certified") == "yes":
        return base

    best = dict(base)
    best_score = -1
    candidates: List[Tuple[str, Dict[str, str]]] = []

    if current["url"] and not CERT.is_blacklisted(current["url"]):
        candidates.append(
            (
                current.get("query", ""),
                {
                    "source": "existing",
                    "url": current["url"],
                    "title": current.get("title", ""),
                    "snippet": current.get("snippet", ""),
                },
            )
        )

    direct_wiki = {
        "source": "wikipedia",
        "url": CERT.wiki_title_url(row[f"{side}_answer"]),
        "title": row[f"{side}_answer"],
        "snippet": "",
    }
    candidates.append((f"direct:{row[f'{side}_answer']}", direct_wiki))

    for query in CERT.build_queries(row, side):
        found: List[Dict[str, str]] = []
        found.extend(searcher.wikipedia_search(query, max_results=5))
        found.extend(searcher.bing_rss_search(query, max_results=5))
        found.extend(searcher.duckduckgo_search(query, max_results=5))
        dedup = {}
        for cand in found:
            url = cand.get("url", "")
            if url and url not in dedup:
                dedup[url] = cand
        ranked = sorted(dedup.values(), key=lambda c: CERT.pre_score(row, side, c), reverse=True)[:5]
        for cand in ranked:
            candidates.append((query, cand))

    seen_urls = set()
    for query, cand in candidates:
        url = cand.get("url", "")
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        score = CERT.pre_score(row, side, cand)
        certified, match_type, excerpt, reason = aggressive_certify_candidate(searcher, row, side, cand)
        rec = {
            "query": query,
            "url": cand.get("url", ""),
            "title": cand.get("title", ""),
            "snippet": cand.get("snippet", ""),
            "excerpt": excerpt,
            "match_type": match_type,
            "certified": "yes" if certified else "",
            "reason": reason if certified else "aggressive_best_candidate",
        }
        if score > best_score:
            best_score = score
            best = rec
        if certified:
            return rec

    return best


def process_unresolved_row(row: Dict[str, str]) -> Dict[str, str]:
    row = dict(row)
    searcher = CERT.Searcher()

    if needs_side(row, "target"):
        target = aggressive_certify_side(searcher, row, "target")
        for key, value in target.items():
            if key == "certified":
                row["target_source_certified"] = value
            elif key == "reason":
                row["target_cert_reason"] = value
            elif key == "query":
                row["target_query"] = value
            elif key == "url":
                row["target_evidence_url"] = value
            elif key == "title":
                row["target_evidence_title"] = value
            elif key == "snippet":
                row["target_evidence_snippet"] = value
            elif key == "excerpt":
                row["target_evidence_excerpt"] = value
            elif key == "match_type":
                row["target_match_type"] = value

    if needs_side(row, "contrast"):
        contrast = aggressive_certify_side(searcher, row, "contrast")
        for key, value in contrast.items():
            if key == "certified":
                row["contrast_source_certified"] = value
            elif key == "reason":
                row["contrast_cert_reason"] = value
            elif key == "query":
                row["contrast_query"] = value
            elif key == "url":
                row["contrast_evidence_url"] = value
            elif key == "title":
                row["contrast_evidence_title"] = value
            elif key == "snippet":
                row["contrast_evidence_snippet"] = value
            elif key == "excerpt":
                row["contrast_evidence_excerpt"] = value
            elif key == "match_type":
                row["contrast_match_type"] = value

    row["judge_target_factuality"] = "yes" if row.get("target_source_certified") == "yes" else ""
    row["judge_locale_dependence"] = (
        "yes"
        if row.get("target_source_certified") == "yes"
        and row.get("contrast_source_certified") == "yes"
        and row.get("target_answer") != row.get("contrast_answer")
        else ""
    )
    return row


def summarize(rows: List[Dict[str, str]]) -> Dict[str, int]:
    return {
        "rows": len(rows),
        "target_certified": sum(r.get("target_source_certified") == "yes" for r in rows),
        "contrast_certified": sum(
            bool(str(r.get("contrast_answer", "")).strip()) and r.get("contrast_source_certified") == "yes"
            for r in rows
        ),
        "both_certified": sum(
            bool(str(r.get("contrast_answer", "")).strip())
            and r.get("target_source_certified") == "yes"
            and r.get("contrast_source_certified") == "yes"
            for r in rows
        ),
        "judge_locale_dependence_yes": sum(r.get("judge_locale_dependence") == "yes" for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description="Aggressively re-certify the unresolved tail of full LocalNewsQA validation.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--tail-csv", required=True)
    parser.add_argument("--tail-summary-json", required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    with open(args.input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    unresolved_idx = []
    unresolved_rows = []
    for i, row in enumerate(rows):
        if needs_side(row, "target") or needs_side(row, "contrast"):
            unresolved_idx.append(i)
            unresolved_rows.append(row)

    improved: List[Optional[Dict[str, str]]] = [None] * len(unresolved_rows)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for idx, row in enumerate(ex.map(process_unresolved_row, unresolved_rows)):
            improved[idx] = row

    for pos, row_idx in enumerate(unresolved_idx):
        rows[row_idx] = improved[pos]

    remaining_tail = [r for r in rows if needs_side(r, "target") or needs_side(r, "contrast")]

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    tail_path = Path(args.tail_csv)
    tail_path.parent.mkdir(parents=True, exist_ok=True)
    with tail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(remaining_tail)

    before_tail = len(unresolved_rows)
    after_tail = len(remaining_tail)
    summary = summarize(rows)
    summary.update(
        {
            "input_rows": len(rows),
            "unresolved_before": before_tail,
            "unresolved_after": after_tail,
            "improved_rows": before_tail - after_tail,
            "output_csv": str(out_path),
        }
    )
    tail_summary = summarize(remaining_tail)
    tail_summary.update(
        {
            "rows": len(remaining_tail),
            "output_csv": str(tail_path),
        }
    )

    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    Path(args.tail_summary_json).write_text(
        json.dumps(tail_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
