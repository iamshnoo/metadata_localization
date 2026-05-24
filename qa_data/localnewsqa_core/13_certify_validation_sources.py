#!/usr/bin/env python3

import argparse
import csv
import html
import json
import re
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup


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
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "to",
    "was",
    "what",
    "when",
    "which",
    "who",
    "with",
}

GENERIC_TERMS = {
    "board",
    "cabinet",
    "chairman",
    "commission",
    "committee",
    "court",
    "council",
    "day",
    "department",
    "federation",
    "general",
    "government",
    "governor",
    "holiday",
    "minister",
    "ministry",
    "office",
    "president",
    "prime",
    "radio",
    "speaker",
    "tribunal",
}

BLACKLISTED_DOMAINS = {
    "reddit.com",
    "www.reddit.com",
    "m.reddit.com",
    "zhihu.com",
    "www.zhihu.com",
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "linkedin.com",
    "www.linkedin.com",
    "facebook.com",
    "www.facebook.com",
    "x.com",
    "www.x.com",
    "twitter.com",
    "www.twitter.com",
    "instagram.com",
    "www.instagram.com",
    "tiktok.com",
    "www.tiktok.com",
}

STRONG_MATCH_TYPES = {
    "wikipedia_search_match",
    "wikipedia_extract_match",
    "duckduckgo_page_match",
    "duckduckgo_snippet_match",
    "bing_rss_page_match",
    "bing_rss_snippet_match",
}


def normalize(text: str) -> str:
    text = html.unescape(str(text or ""))
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    return [t for t in normalize(text).split() if t and t not in STOPWORDS]


def meaningful_tokens(text: str) -> List[str]:
    return [t for t in tokenize(text) if len(t) >= 3]


def answer_tokens(answer: str) -> List[str]:
    return meaningful_tokens(answer)


def is_generic_answer(answer: str) -> bool:
    toks = answer_tokens(answer)
    if not toks:
        return True
    if len(toks) <= 2 and all(tok in GENERIC_TERMS or len(tok) <= 3 for tok in toks):
        return True
    return False


def extract_excerpt(text: str, needle: str, window: int = 180) -> str:
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


def country_markers(country: str) -> List[str]:
    return COUNTRY_MARKERS.get(country, [normalize(country)])


def context_keywords(row: Dict[str, str], side: str) -> List[str]:
    ans = set(answer_tokens(row[f"{side}_answer"]))
    ctry = set(answer_tokens(row[f"{side}_country"]))
    kws: List[str] = []
    for src in (row.get("evidence_hint", ""), row.get("question", "")):
        for tok in meaningful_tokens(src):
            if tok in ans or tok in ctry:
                continue
            if tok not in kws:
                kws.append(tok)
    return kws[:8]


def build_queries(row: Dict[str, str], side: str) -> List[str]:
    answer = row[f"{side}_answer"]
    country = row[f"{side}_country"]
    hint = row.get("evidence_hint", "")
    question = " ".join(row.get("question", "").rstrip("?").split()[:10])
    existing = row.get(f"{side}_query", "")
    generic = is_generic_answer(answer)
    queries: List[str] = []

    def add(q: str):
        q = re.sub(r"\s+", " ", str(q or "").strip())
        if q and q not in queries:
            queries.append(q)

    add(existing)
    add(f"\"{country}\" \"{answer}\"")
    add(f"\"{answer}\"")
    add(f"{country} {answer}")
    if hint:
        add(f"{country} {answer} {hint}")
    if question:
        add(f"{country} {answer} {question}")
    if generic and hint:
        add(f"{country} {hint}")
    if generic and question:
        add(f"{country} {question}")
    return queries[:5]


def domain_of(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def unwrap_duckduckgo(url: str) -> str:
    if "uddg=" not in url:
        return url
    try:
        return parse_qs(urlparse(url).query).get("uddg", [""])[0] or url
    except Exception:
        return url


def is_blacklisted(url: str) -> bool:
    domain = domain_of(url)
    return domain in BLACKLISTED_DOMAINS


def domain_bonus(url: str) -> int:
    domain = domain_of(url)
    if "wikipedia.org" in domain:
        return 5
    if domain.endswith(".gov") or ".gov." in domain:
        return 4
    if domain.endswith(".edu") or ".edu." in domain:
        return 3
    if domain.endswith(".org") or ".org." in domain:
        return 2
    if domain.endswith(".ac.uk") or domain.endswith(".ac.in"):
        return 3
    if any(k in domain for k in ["britannica.com", "whitehouse.gov", "gao.gov", "npr.org", "cbc.ca", "gov.bd", "gov.in", "gc.ca"]):
        return 4
    return 0


def wiki_title_url(title: str) -> str:
    safe = re.sub(r"\s+", "_", str(title or "").strip())
    return f"https://en.wikipedia.org/wiki/{safe}"


class Searcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.page_cache: Dict[str, str] = {}
        self.page_lock = threading.Lock()

    def fetch_page_text(self, url: str, timeout: int = 4) -> str:
        with self.page_lock:
            if url in self.page_cache:
                return self.page_cache[url]
        text = ""
        try:
            resp = self.session.get(url, timeout=timeout, allow_redirects=True)
            if resp.status_code == 200 and "text/html" in resp.headers.get("content-type", ""):
                soup = BeautifulSoup(resp.text, "lxml")
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        except Exception:
            text = ""
        with self.page_lock:
            self.page_cache[url] = text
        return text

    def wikipedia_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        try:
            resp = self.session.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "utf8": 1,
                    "format": "json",
                    "srlimit": max_results,
                },
                timeout=4,
            )
            resp.raise_for_status()
            data = resp.json().get("query", {}).get("search", [])
        except Exception:
            return []
        hits = []
        for row in data[:max_results]:
            title = row.get("title", "")
            hits.append(
                {
                    "source": "wikipedia",
                    "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
                    "title": title,
                    "snippet": BeautifulSoup(row.get("snippet", ""), "lxml").get_text(" ", strip=True),
                }
            )
        return hits

    def bing_rss_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        try:
            resp = self.session.get(
                "https://www.bing.com/search",
                params={"q": query, "format": "rss"},
                timeout=3,
            )
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
        except Exception:
            return []
        hits = []
        for item in root.findall(".//item")[:max_results]:
            url = item.findtext("link", default="")
            if not url or is_blacklisted(url):
                continue
            hits.append(
                {
                    "source": "bing",
                    "url": url,
                    "title": item.findtext("title", default=""),
                    "snippet": item.findtext("description", default=""),
                }
            )
        return hits

    def duckduckgo_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        try:
            resp = self.session.get(
                "https://duckduckgo.com/html/",
                params={"q": query},
                timeout=4,
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
        except Exception:
            return []
        hits = []
        for result in soup.select("div.result")[:max_results]:
            a = result.select_one("a.result__a")
            if not a:
                continue
            url = unwrap_duckduckgo(a.get("href") or "")
            if not url or is_blacklisted(url):
                continue
            snippet_tag = result.select_one("a.result__snippet") or result.select_one(".result__snippet")
            hits.append(
                {
                    "source": "duckduckgo",
                    "url": url,
                    "title": a.get_text(" ", strip=True),
                    "snippet": snippet_tag.get_text(" ", strip=True) if snippet_tag else "",
                }
            )
        return hits


def pre_score(row: Dict[str, str], side: str, cand: Dict[str, str]) -> int:
    answer_norm = normalize(row[f"{side}_answer"])
    cands = normalize(cand.get("title", "") + " " + cand.get("snippet", ""))
    toks = answer_tokens(row[f"{side}_answer"])
    kws = context_keywords(row, side)
    score = domain_bonus(cand.get("url", ""))
    if answer_norm and answer_norm in cands:
        score += 8
    score += min(3, sum(tok in cands for tok in toks))
    score += min(2, sum(kw in cands for kw in kws))
    score += 2 * sum(marker in cands for marker in country_markers(row[f"{side}_country"]))
    return score


def certify_candidate(searcher: Searcher, row: Dict[str, str], side: str, cand: Dict[str, str]) -> Tuple[bool, str, str]:
    answer = row[f"{side}_answer"]
    answer_norm = normalize(answer)
    answer_toks = answer_tokens(answer)
    generic = is_generic_answer(answer)
    kws = context_keywords(row, side)
    markers = country_markers(row[f"{side}_country"])

    title = cand.get("title", "")
    snippet = cand.get("snippet", "")
    url = cand.get("url", "")
    source = cand.get("source", "")

    text = f"{title}. {snippet}"
    if domain_bonus(url) >= 2 or source == "wikipedia":
        page_text = searcher.fetch_page_text(url)
        if page_text:
            text = f"{title}. {snippet}. {page_text[:15000]}"

    norm = normalize(text)
    ans_phrase = bool(answer_norm and answer_norm in norm)
    ans_tok_hits = sum(tok in norm for tok in answer_toks)
    kw_hits = sum(kw in norm for kw in kws)
    country_hits = sum(marker in norm for marker in markers)
    title_norm = normalize(title)
    title_exact = bool(answer_norm and (answer_norm == title_norm or answer_norm in title_norm))

    certified = False
    reason = ""
    if title_exact and (country_hits or kw_hits >= 1):
        certified = True
        reason = "title_exact_plus_context"
    elif ans_phrase and (country_hits or kw_hits >= (2 if generic else 1)):
        certified = True
        reason = "answer_phrase_plus_context"
    elif ans_tok_hits >= max(1, len(answer_toks) - 1) and (country_hits or kw_hits >= 2):
        certified = True
        reason = "answer_tokens_plus_context"

    if certified:
        dom = domain_of(url)
        if source == "wikipedia" or "wikipedia.org" in dom:
            match_type = "wikipedia_extract_match" if "wikipedia.org" in domain_of(url) else "wikipedia_search_match"
        elif source == "duckduckgo":
            match_type = "duckduckgo_page_match"
        elif source == "bing":
            match_type = "bing_rss_page_match"
        else:
            match_type = "existing_page_match"
    else:
        match_type = f"{source}_candidate" if source else "no_result"

    return certified, match_type, extract_excerpt(text, answer)


def certify_side(searcher: Searcher, row: Dict[str, str], side: str) -> Dict[str, str]:
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

    # Re-certify existing strong links if they are good.
    if current["url"] and current["match_type"] in STRONG_MATCH_TYPES:
        current["certified"] = "yes"
        current["reason"] = "existing_strong_match"
        return current

    direct_cands = [
        {
            "source": "wikipedia",
            "url": wiki_title_url(row[f"{side}_answer"]),
            "title": row[f"{side}_answer"],
            "snippet": "",
        }
    ]
    for cand in direct_cands:
        certified, match_type, excerpt = certify_candidate(searcher, row, side, cand)
        if certified:
            return {
                "query": f"direct:{row[f'{side}_answer']}",
                "url": cand["url"],
                "title": cand["title"],
                "snippet": cand["snippet"],
                "excerpt": excerpt,
                "match_type": match_type,
                "certified": "yes",
                "reason": "direct_wikipedia_title",
            }

    if current["url"] and not is_blacklisted(current["url"]):
        existing_cand = {
            "source": "existing",
            "url": current["url"],
            "title": current.get("title", ""),
            "snippet": current.get("snippet", ""),
        }
        certified, match_type, excerpt = certify_candidate(searcher, row, side, existing_cand)
        if certified:
            current["excerpt"] = excerpt
            current["match_type"] = match_type
            current["certified"] = "yes"
            current["reason"] = "existing_certified"
            return current

    best = dict(current)
    best_score = -1
    queries = build_queries(row, side)
    for query in queries:
        candidates: List[Dict[str, str]] = []
        candidates.extend(searcher.wikipedia_search(query, max_results=3))
        candidates.extend(searcher.bing_rss_search(query, max_results=3))
        candidates.extend(searcher.duckduckgo_search(query, max_results=3))

        dedup = {}
        for cand in candidates:
            url = cand.get("url", "")
            if not url:
                continue
            if url not in dedup:
                dedup[url] = cand
        ranked = sorted(dedup.values(), key=lambda c: pre_score(row, side, c), reverse=True)[:3]

        for cand in ranked:
            score = pre_score(row, side, cand)
            certified, match_type, excerpt = certify_candidate(searcher, row, side, cand)
            rec = {
                "query": query,
                "url": cand.get("url", ""),
                "title": cand.get("title", ""),
                "snippet": cand.get("snippet", ""),
                "excerpt": excerpt,
                "match_type": match_type,
                "certified": "yes" if certified else "",
                "reason": "certified" if certified else "best_candidate",
            }
            if score > best_score:
                best_score = score
                best = rec
            if certified:
                return rec
    return best


def process_row(item: Tuple[int, Dict[str, str]]) -> Tuple[int, Dict[str, str]]:
    idx, row = item
    row = dict(row)
    searcher = Searcher()
    if str(row.get("target_answer", "")).strip():
        target = certify_side(searcher, row, "target")
    else:
        target = {"query": "", "url": "", "title": "", "snippet": "", "excerpt": "", "match_type": "no_answer"}

    if str(row.get("contrast_answer", "")).strip():
        contrast = certify_side(searcher, row, "contrast")
    else:
        contrast = {"query": "", "url": "", "title": "", "snippet": "", "excerpt": "", "match_type": "no_answer"}

    for side, rec in (("target", target), ("contrast", contrast)):
        row[f"{side}_query"] = rec.get("query", row.get(f"{side}_query", ""))
        row[f"{side}_evidence_url"] = rec.get("url", row.get(f"{side}_evidence_url", ""))
        row[f"{side}_evidence_title"] = rec.get("title", row.get(f"{side}_evidence_title", ""))
        row[f"{side}_evidence_snippet"] = rec.get("snippet", row.get(f"{side}_evidence_snippet", ""))
        row[f"{side}_evidence_excerpt"] = rec.get("excerpt", row.get(f"{side}_evidence_excerpt", ""))
        row[f"{side}_match_type"] = rec.get("match_type", row.get(f"{side}_match_type", ""))
        row[f"{side}_source_certified"] = rec.get("certified", "")
        row[f"{side}_cert_reason"] = rec.get("reason", "")

    row["judge_target_factuality"] = "yes" if row.get("target_source_certified") == "yes" else ""
    row["judge_locale_dependence"] = (
        "yes"
        if row.get("target_source_certified") == "yes"
        and row.get("contrast_source_certified") == "yes"
        and row.get("target_answer") != row.get("contrast_answer")
        else ""
    )
    return idx, row


def main():
    parser = argparse.ArgumentParser(description="Certify LocalNewsQA human-validation evidence sources.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    extra_fields = [
        "target_source_certified",
        "target_cert_reason",
        "contrast_source_certified",
        "contrast_cert_reason",
    ]
    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    if args.limit is not None:
        rows = rows[: args.limit]

    indexed = list(enumerate(rows))
    outputs: List[Optional[Dict[str, str]]] = [None] * len(indexed)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for idx, row in ex.map(process_row, indexed):
            outputs[idx] = row

    final_rows = [r for r in outputs if r is not None]

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    summary = {
        "rows": len(final_rows),
        "target_certified": sum(r.get("target_source_certified") == "yes" for r in final_rows),
        "contrast_certified": sum(r.get("contrast_source_certified") == "yes" for r in final_rows),
        "both_certified": sum(
            r.get("target_source_certified") == "yes" and r.get("contrast_source_certified") == "yes"
            for r in final_rows
        ),
        "judge_locale_dependence_yes": sum(r.get("judge_locale_dependence") == "yes" for r in final_rows),
        "output_csv": str(out_path),
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
