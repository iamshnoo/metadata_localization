#!/usr/bin/env python3

import argparse
import csv
import html
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
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
    "led",
    "local",
    "main",
    "might",
    "news",
    "of",
    "on",
    "or",
    "report",
    "says",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "which",
    "who",
    "with",
    "would",
}

GENERIC_ANSWER_TERMS = {
    "assembly",
    "cabinet",
    "chief",
    "council",
    "court",
    "crown",
    "department",
    "general",
    "governor",
    "justice",
    "minister",
    "parliament",
    "police",
    "president",
    "prime",
    "radio",
    "senate",
    "speaker",
    "supreme",
    "tribunal",
}

STRONG_MATCH_TYPES = {
    "wikipedia_search_match",
    "wikipedia_extract_match",
    "duckduckgo_snippet_match",
    "duckduckgo_page_match",
}


def normalize(text: str) -> str:
    text = html.unescape(str(text or ""))
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_excerpt(text: str, needle: str, window: int = 160) -> str:
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


def clean_tokens(text: str) -> List[str]:
    toks = normalize(text).split()
    return [t for t in toks if len(t) >= 4 and t not in STOPWORDS]


def is_generic_answer(answer: str) -> bool:
    toks = clean_tokens(answer)
    if not toks:
        return True
    if len(toks) == 1 and toks[0] in GENERIC_ANSWER_TERMS:
        return True
    if len(toks) <= 2 and all(t in GENERIC_ANSWER_TERMS for t in toks):
        return True
    return False


def context_keywords(question: str, evidence_hint: str, country: str, answer: str) -> List[str]:
    blocked = set(clean_tokens(country)) | set(clean_tokens(answer))
    kws: List[str] = []
    for src in (evidence_hint, question):
        for tok in clean_tokens(src):
            if tok not in blocked and tok not in kws:
                kws.append(tok)
    return kws[:8]


def build_queries(
    question: str,
    country: str,
    answer: str,
    evidence_hint: str,
    existing_query: str = "",
) -> List[str]:
    queries: List[str] = []

    def add(q: str):
        q = re.sub(r"\s+", " ", str(q or "").strip())
        if q and q not in queries:
            queries.append(q)

    question_keywords = " ".join(str(question).strip().rstrip("?").split()[:10])
    generic = is_generic_answer(answer)

    add(existing_query)
    add(f"\"{country}\" \"{answer}\"")
    add(f"{country} {answer}")
    if evidence_hint:
        add(f"\"{country}\" \"{answer}\" {evidence_hint}")
        add(f"{country} {answer} {evidence_hint}")
    if question_keywords:
        add(f"{country} {answer} {question_keywords}")
        add(f"{country} {question_keywords}")
    if evidence_hint and question_keywords:
        add(f"{country} {evidence_hint} {question_keywords}")
        add(f"{answer} {evidence_hint} {question_keywords}")
    if generic:
        add(f"{country} {evidence_hint}")
        add(f"{country} {question_keywords}")
    return queries[:5]


class EvidenceSearcher:
    def __init__(self, sleep_seconds: float = 0.4, max_results: int = 5):
        self.sleep_seconds = sleep_seconds
        self.max_results = max_results
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.search_cache: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
        self.page_cache: Dict[str, str] = {}
        self.wiki_extract_cache: Dict[str, str] = {}

    def fetch_page_text(self, url: str, timeout: int = 8) -> str:
        if url in self.page_cache:
            return self.page_cache[url]
        try:
            resp = self.session.get(url, timeout=timeout, allow_redirects=True)
            if resp.status_code != 200 or "text/html" not in resp.headers.get("content-type", ""):
                self.page_cache[url] = ""
                return ""
            soup = BeautifulSoup(resp.text, "lxml")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
            self.page_cache[url] = text
            return text
        except Exception:
            self.page_cache[url] = ""
            return ""

    def wikipedia_search(self, query: str) -> List[Dict[str, str]]:
        cache_key = ("wikipedia", query)
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        try:
            resp = self.session.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "utf8": 1,
                    "format": "json",
                    "srlimit": self.max_results,
                },
                timeout=8,
            )
            resp.raise_for_status()
            data = resp.json().get("query", {}).get("search", [])
        except Exception:
            data = []
        hits = []
        for row in data[: self.max_results]:
            title = row.get("title", "")
            hits.append(
                {
                    "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
                    "title": title,
                    "snippet": BeautifulSoup(row.get("snippet", ""), "lxml").get_text(" ", strip=True),
                    "source": "wikipedia",
                }
            )
        self.search_cache[cache_key] = hits
        return hits

    def wikipedia_extract(self, title: str) -> str:
        if title in self.wiki_extract_cache:
            return self.wiki_extract_cache[title]
        try:
            resp = self.session.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": 1,
                    "titles": title,
                    "format": "json",
                },
                timeout=8,
            )
            resp.raise_for_status()
            pages = resp.json().get("query", {}).get("pages", {})
            extract = ""
            for page in pages.values():
                extract = re.sub(r"\s+", " ", page.get("extract", ""))
                if extract:
                    break
        except Exception:
            extract = ""
        self.wiki_extract_cache[title] = extract
        return extract

    def duckduckgo_search(self, query: str) -> List[Dict[str, str]]:
        cache_key = ("duckduckgo", query)
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        try:
            resp = self.session.get(
                "https://duckduckgo.com/html/",
                params={"q": query},
                timeout=8,
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
        except Exception:
            self.search_cache[cache_key] = []
            return []
        hits = []
        for result in soup.select("div.result")[: self.max_results]:
            a = result.select_one("a.result__a")
            if not a:
                continue
            url = a.get("href") or ""
            if "uddg=" in url:
                url = parse_qs(urlparse(url).query).get("uddg", [""])[0]
            snippet_tag = result.select_one("a.result__snippet") or result.select_one(".result__snippet")
            hits.append(
                {
                    "url": url,
                    "title": a.get_text(" ", strip=True),
                    "snippet": snippet_tag.get_text(" ", strip=True) if snippet_tag else "",
                    "source": "duckduckgo",
                }
            )
        self.search_cache[cache_key] = hits
        return hits

    def bing_rss_search(self, query: str) -> List[Dict[str, str]]:
        cache_key = ("bing", query)
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        try:
            resp = self.session.get(
                "https://www.bing.com/search",
                params={"q": query, "format": "rss"},
                timeout=6,
            )
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
        except Exception:
            self.search_cache[cache_key] = []
            return []
        hits = []
        for item in root.findall(".//item")[: self.max_results]:
            hits.append(
                {
                    "url": item.findtext("link", default=""),
                    "title": item.findtext("title", default=""),
                    "snippet": item.findtext("description", default=""),
                    "source": "bing",
                }
            )
        self.search_cache[cache_key] = hits
        return hits

    def evaluate_candidate(
        self,
        row: Dict[str, str],
        answer: str,
        country: str,
        candidate: Dict[str, str],
        rank: int,
    ) -> Dict[str, str]:
        answer_norm = normalize(answer)
        title = candidate.get("title", "")
        snippet = candidate.get("snippet", "")
        title_norm = normalize(title)
        snippet_norm = normalize(snippet)
        country_markers = COUNTRY_MARKERS.get(country, [normalize(country)])
        generic = is_generic_answer(answer)
        keywords = context_keywords(
            row.get("question", ""),
            row.get("evidence_hint", ""),
            country,
            answer,
        )

        answer_in_title = bool(answer_norm and answer_norm in title_norm)
        answer_in_snippet = bool(answer_norm and answer_norm in snippet_norm)
        country_in_title = any(marker in title_norm for marker in country_markers)
        country_in_snippet = any(marker in snippet_norm for marker in country_markers)

        combined_pre = " ".join(x for x in [title_norm, snippet_norm] if x)
        kw_hits_pre = [kw for kw in keywords if kw in combined_pre]

        page_text = ""
        page_norm = ""
        answer_in_page = False
        country_in_page = False
        kw_hits_page: List[str] = []
        if candidate.get("source") == "wikipedia" and title:
            page_text = self.wikipedia_extract(title)
            page_norm = normalize(page_text)
            answer_in_page = bool(answer_norm and answer_norm in page_norm)
            country_in_page = any(marker in page_norm for marker in country_markers)
            kw_hits_page = [kw for kw in keywords if kw in page_norm]

        kw_hits = kw_hits_pre[:]
        for kw in kw_hits_page:
            if kw not in kw_hits:
                kw_hits.append(kw)

        score = 0
        if answer_in_title:
            score += 7
        if answer_in_snippet:
            score += 5
        if answer_in_page:
            score += 4
        if country_in_title:
            score += 3
        if country_in_snippet:
            score += 2
        if country_in_page:
            score += 2
        score += min(3, len(kw_hits))

        if rank == 1:
            score += 1

        if candidate.get("source") == "duckduckgo" and candidate.get("url", "").endswith((".gov", ".gov/")):
            score += 1

        source = candidate.get("source", "")
        match_type = "no_result"
        strong = False
        if answer_in_title or answer_in_snippet:
            if not generic and (country_in_title or country_in_snippet or country_in_page):
                match_type = (
                    "wikipedia_search_match" if source == "wikipedia" else "duckduckgo_snippet_match"
                )
                strong = True
            elif generic and kw_hits and (country_in_title or country_in_snippet or country_in_page):
                match_type = (
                    "wikipedia_search_match" if source == "wikipedia" else "duckduckgo_snippet_match"
                )
                strong = True
        if not strong and answer_in_page:
            if not generic and (country_in_page or country_in_title or country_in_snippet):
                match_type = (
                    "wikipedia_extract_match" if source == "wikipedia" else "duckduckgo_page_match"
                )
                strong = True
            elif generic and kw_hits and (country_in_page or country_in_title or country_in_snippet):
                match_type = (
                    "wikipedia_extract_match" if source == "wikipedia" else "duckduckgo_page_match"
                )
                strong = True

        if not strong and score >= 5 and candidate.get("url"):
            match_type = "wikipedia_candidate" if source == "wikipedia" else "duckduckgo_candidate"

        excerpt_source = page_text or f"{title}. {snippet}"
        excerpt = extract_excerpt(excerpt_source, answer) if answer else excerpt_source[:320]
        return {
            "query": "",
            "url": candidate.get("url", ""),
            "title": title,
            "snippet": snippet,
            "excerpt": excerpt,
            "match_type": match_type,
            "error": "",
            "rank": str(rank),
            "score": score,
            "strong": strong,
        }

    def search_evidence(
        self,
        row: Dict[str, str],
        answer_key: str,
        country_key: str,
        query_key: str,
    ) -> Dict[str, str]:
        answer = row.get(answer_key, "")
        country = row.get(country_key, "")
        queries = build_queries(
            row.get("question", ""),
            country,
            answer,
            row.get("evidence_hint", ""),
            row.get(query_key, ""),
        )
        fallback = {
            "query": queries[0] if queries else "",
            "url": "",
            "title": "",
            "snippet": "",
            "excerpt": "",
            "match_type": "no_result",
            "error": "",
        }
        best = dict(fallback)
        best_score = -1
        for query in queries:
            for source_name, search_fn in (
                ("wikipedia", self.wikipedia_search),
                ("bing", self.bing_rss_search),
                ("duckduckgo", self.duckduckgo_search),
            ):
                try:
                    candidates = search_fn(query)
                except Exception as e:
                    fallback["error"] = f"{type(e).__name__}: {e}"
                    continue
                for rank, candidate in enumerate(candidates, start=1):
                    record = self.evaluate_candidate(row, answer, country, candidate, rank)
                    record["query"] = query
                    if record["score"] > best_score:
                        best_score = record["score"]
                        best = record
                    if record["match_type"] in STRONG_MATCH_TYPES:
                        return record
                    if record["match_type"].endswith("candidate") and record["score"] >= 10:
                        return record
                if candidates:
                    time.sleep(self.sleep_seconds)
            time.sleep(self.sleep_seconds)
        return best


def rewrite_row(row: Dict[str, str], target_ev: Dict[str, str], contrast_ev: Dict[str, str]) -> Dict[str, str]:
    out = dict(row)
    for prefix, ev in (("target", target_ev), ("contrast", contrast_ev)):
        out[f"{prefix}_query"] = ev.get("query", out.get(f"{prefix}_query", ""))
        out[f"{prefix}_evidence_url"] = ev.get("url", "")
        out[f"{prefix}_evidence_title"] = ev.get("title", "")
        out[f"{prefix}_evidence_snippet"] = ev.get("snippet", "")
        out[f"{prefix}_evidence_excerpt"] = ev.get("excerpt", "")
        out[f"{prefix}_match_type"] = ev.get("match_type", "")

    no_leakage = str(out.get("judge_no_explicit_leakage", "")).strip() or "yes"
    target_ok = out.get("target_match_type", "") in STRONG_MATCH_TYPES
    contrast_ok = out.get("contrast_match_type", "") in STRONG_MATCH_TYPES
    locale_dep = target_ok and contrast_ok and out.get("target_answer") != out.get("contrast_answer")

    out["judge_target_factuality"] = "yes" if target_ok else ""
    out["judge_locale_dependence"] = "yes" if locale_dep else ""
    out["judge_no_explicit_leakage"] = no_leakage
    out["annotator_notes"] = (
        f"target={out.get('target_country','')}:{out.get('target_answer','')} "
        f"[{out.get('target_match_type','')}]; "
        f"contrast={out.get('contrast_country','')}:{out.get('contrast_answer','')} "
        f"[{out.get('contrast_match_type','')}]"
    )
    return out


def write_csv(path: Path, rows: Iterable[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Fill missing evidence URLs for LocalNewsQA human-validation rows.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--progress-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--sleep-seconds", type=float, default=0.4)
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--blank-only", action="store_true")
    args = parser.parse_args()

    with open(args.input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if args.limit is not None:
        rows = rows[: args.limit]
    rows = rows[args.start_index : args.end_index]

    searcher = EvidenceSearcher(
        sleep_seconds=args.sleep_seconds,
        max_results=args.max_results,
    )

    outputs: List[Dict[str, str]] = []
    before_target = sum(bool((r.get("target_evidence_url") or "").strip()) for r in rows)
    before_contrast = sum(bool((r.get("contrast_evidence_url") or "").strip()) for r in rows)

    progress_path = Path(args.progress_jsonl)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("w", encoding="utf-8") as progress_f:
        for idx, row in enumerate(rows, start=1):
            target_ev = {
                "query": row.get("target_query", ""),
                "url": row.get("target_evidence_url", ""),
                "title": row.get("target_evidence_title", ""),
                "snippet": row.get("target_evidence_snippet", ""),
                "excerpt": row.get("target_evidence_excerpt", ""),
                "match_type": row.get("target_match_type", ""),
            }
            contrast_ev = {
                "query": row.get("contrast_query", ""),
                "url": row.get("contrast_evidence_url", ""),
                "title": row.get("contrast_evidence_title", ""),
                "snippet": row.get("contrast_evidence_snippet", ""),
                "excerpt": row.get("contrast_evidence_excerpt", ""),
                "match_type": row.get("contrast_match_type", ""),
            }

            should_search_target = (
                not target_ev["url"] if args.blank_only else target_ev["match_type"] not in STRONG_MATCH_TYPES
            )
            should_search_contrast = (
                not contrast_ev["url"] if args.blank_only else contrast_ev["match_type"] not in STRONG_MATCH_TYPES
            )

            if should_search_target:
                searched = searcher.search_evidence(row, "target_answer", "target_country", "target_query")
                if searched.get("url"):
                    target_ev = searched
            if should_search_contrast:
                searched = searcher.search_evidence(row, "contrast_answer", "contrast_country", "contrast_query")
                if searched.get("url"):
                    contrast_ev = searched

            out_row = rewrite_row(row, target_ev, contrast_ev)
            outputs.append(out_row)
            progress_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            progress_f.flush()

            if idx % 25 == 0:
                print(f"Processed {idx}/{len(rows)}")

    write_csv(Path(args.output_csv), outputs, fieldnames)

    after_target = sum(bool((r.get("target_evidence_url") or "").strip()) for r in outputs)
    after_contrast = sum(bool((r.get("contrast_evidence_url") or "").strip()) for r in outputs)
    after_either = sum(
        bool((r.get("target_evidence_url") or "").strip() or (r.get("contrast_evidence_url") or "").strip())
        for r in outputs
    )
    strong_target = sum(r.get("target_match_type", "") in STRONG_MATCH_TYPES for r in outputs)
    strong_contrast = sum(r.get("contrast_match_type", "") in STRONG_MATCH_TYPES for r in outputs)
    locale_yes = sum(r.get("judge_locale_dependence", "") == "yes" for r in outputs)

    summary = {
        "input_csv": args.input_csv,
        "output_csv": args.output_csv,
        "progress_jsonl": args.progress_jsonl,
        "rows": len(outputs),
        "before_target_urls": before_target,
        "before_contrast_urls": before_contrast,
        "after_target_urls": after_target,
        "after_contrast_urls": after_contrast,
        "after_either_urls": after_either,
        "strong_target_matches": strong_target,
        "strong_contrast_matches": strong_contrast,
        "judge_locale_dependence_yes": locale_yes,
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
