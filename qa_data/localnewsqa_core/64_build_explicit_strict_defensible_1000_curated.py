#!/usr/bin/env python3

import csv
import importlib.util
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import requests


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_semantic_gold_ambiguous_1700"
AUDIT_DIR = BASE / "explicit_max_audit"
SOURCE_STRICT = AUDIT_DIR / "strict_defensible_954/localnewsqa_targetqa_explicit_strict_defensible_954_per_country.jsonl"
AMBIGUOUS = BASE / "localnewsqa_ambiguous_semantic_gold_1700.jsonl"
OUTDIR = AUDIT_DIR / "strict_defensible_1000_curated"
OUT_JSONL = OUTDIR / "localnewsqa_targetqa_explicit_strict_defensible_1000_per_country_curated.jsonl"
OUT_CSV = OUTDIR / "localnewsqa_targetqa_explicit_strict_defensible_1000_per_country_curated.csv"
SUMMARY = OUTDIR / "strict_defensible_1000_curated_build_summary.json"
CURATION_LOG = OUTDIR / "curated_additions_log.csv"
REJECT_LOG = OUTDIR / "curated_candidate_reject_log.csv"
FETCH_CACHE = OUTDIR / "strict_defensible_1000_curated_target_evidence_fetches.jsonl"
CANDIDATE_FETCH_CACHE = OUTDIR / "curated_candidate_wikipedia_fetches.jsonl"

CANDIDATE_POOLS = [
    AUDIT_DIR / "strict_1000/localnewsqa_targetqa_explicit_strict_1000_per_country_polished.jsonl",
    AUDIT_DIR / "localnewsqa_targetqa_explicit_style_max_strict_no_warnings.jsonl",
    AUDIT_DIR / "localnewsqa_targetqa_explicit_style_max_paper_clean.jsonl",
    AUDIT_DIR / "localnewsqa_targetqa_explicit_style_max_clean.jsonl",
]
FETCH_CACHES = [
    BASE / "semantic_gold_selected_evidence_fetches.jsonl",
    AUDIT_DIR / "explicit_target_evidence_fetches.jsonl",
    AUDIT_DIR / "strict_1000/polished_audit/explicit_target_evidence_fetches.jsonl",
    AUDIT_DIR / "strict_defensible_954/strict_defensible_target_evidence_fetches.jsonl",
    AUDIT_DIR / "strict_defensible_954/audit/explicit_target_evidence_fetches.jsonl",
    FETCH_CACHE,
    CANDIDATE_FETCH_CACHE,
]

AUDIT_EXPLICIT_SCRIPT = ROOT / "qa_data/localnewsqa_core/48_audit_explicit_max_split.py"
FETCH_AUDIT_SCRIPT = ROOT / "qa_data/localnewsqa_core/32_web_audit_ambiguous_verifiable.py"
BUILDER_SCRIPT = ROOT / "qa_data/localnewsqa_core/45_build_relation_strict_gold_ambiguous.py"

COUNTRY_ORDER = [
    "Bangladesh",
    "Canada",
    "Ghana",
    "Hong Kong",
    "India",
    "Ireland",
    "Jamaica",
    "Kenya",
    "Malaysia",
    "Nigeria",
    "Pakistan",
    "Philippines",
    "South Africa",
    "Sri Lanka",
    "Tanzania",
    "United Kingdom",
    "United States",
]

COUNTRY_MARKERS = {
    "Bangladesh": ["bangladesh", "bangladeshi"],
    "Canada": ["canada", "canadian"],
    "Ghana": ["ghana", "ghanaian"],
    "Hong Kong": ["hong kong", "hongkonger", "hong konger"],
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
    "United Kingdom": ["united kingdom", "uk", "british", "england", "scotland", "wales"],
    "United States": ["united states", "u s", "us", "american"],
}

SOURCE_PRIORITY = {
    "explicit": 0,
    "ambiguous_salvaged_target": 1,
    "ambiguous_pool_salvaged_target_web_supported": 2,
}

BAD_TEXT_FRAGMENTS = {
    "may refer to",
    "can refer to",
    "topics referred to by the same term",
    "see also",
    "external links",
    "references",
    "http://",
    "https://",
    "find sources",
    "edit links",
    "toggle",
    "jump to",
}

BAD_QUESTION_FRAGMENTS = {
    "which answer",
    "associated with",
    "the answer",
    "http",
    "see also",
    "may refer",
    "topics referred",
    "edit links",
    "toggle",
}

PERSON_WORDS = {
    "actor",
    "actress",
    "activist",
    "artist",
    "athlete",
    "businessman",
    "businesswoman",
    "composer",
    "cricketer",
    "director",
    "entrepreneur",
    "filmmaker",
    "footballer",
    "journalist",
    "minister",
    "model",
    "musician",
    "painter",
    "personality",
    "player",
    "politician",
    "producer",
    "rapper",
    "singer",
    "songwriter",
    "writer",
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


def norm(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def rank(row: dict) -> tuple:
    return (
        SOURCE_PRIORITY.get(row.get("source_split_type", ""), 9),
        norm(row.get("target_answer", "")),
        norm(row.get("question", "")),
        row.get("source_row_id", ""),
    )


def wiki_title_from_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc not in {"en.wikipedia.org", "en.m.wikipedia.org"} or not parsed.path.startswith("/wiki/"):
        return ""
    return unquote(parsed.path[len("/wiki/") :]).replace("_", " ").strip()


def needs_wikipedia_api_hydration(url: str, fetch: dict) -> bool:
    if not wiki_title_from_url(url):
        return False
    text = str(fetch.get("text", "") or "")
    content_type = str(fetch.get("content_type", "") or "")
    if not fetch.get("ok"):
        return True
    if "application/json" in content_type and "jump to content" not in text[:1200].lower():
        return False
    return "jump to content" in text[:2000].lower() or "from wikipedia, the free encyclopedia" in text[:2000].lower()


def fetch_wikipedia_leads(urls: list[str]) -> dict[str, dict]:
    pairs = [(url, wiki_title_from_url(url)) for url in urls if wiki_title_from_url(url)]
    out: dict[str, dict] = {}
    if not pairs:
        return out
    api_url = "https://en.wikipedia.org/w/api.php"
    batch_size = 10
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
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
                    "titles": "|".join(title for _, title in batch),
                },
                timeout=30,
                headers={"User-Agent": "LocalNewsQA-strict-curation/1.0"},
            )
            response.raise_for_status()
            data = response.json().get("query", {})
        except Exception as exc:
            for url, title in batch:
                out[url] = {
                    "url": url,
                    "ok": False,
                    "status_code": "",
                    "final_url": "",
                    "content_type": "application/json",
                    "title": title,
                    "text": "",
                    "text_len": 0,
                    "error": f"wikipedia_lead_fetch_failed:{str(exc)[:200]}",
                    "elapsed_sec": 0.0,
                }
            continue
        redirects = {item.get("from", ""): item.get("to", "") for item in data.get("redirects", [])}
        normalized = {item.get("from", ""): item.get("to", "") for item in data.get("normalized", [])}
        pages = {page.get("title", ""): page for page in data.get("pages", {}).values()}
        for url, original_title in batch:
            title = redirects.get(original_title, normalized.get(original_title, original_title))
            title = redirects.get(title, normalized.get(title, title))
            page = pages.get(title) or pages.get(original_title)
            text = re.sub(r"\s+", " ", str(page.get("extract", "") if page else "")).strip()
            out[url] = {
                "url": url,
                "ok": bool(text),
                "status_code": "200" if text else "404",
                "final_url": page.get("fullurl", url) if page else "",
                "content_type": "application/json",
                "title": page.get("title", original_title) if page else original_title,
                "text": text,
                "text_len": len(text),
                "error": "" if text else "wikipedia_lead_empty",
                "elapsed_sec": 0.0,
            }
        print(f"wikipedia full-text hydrated {min(start + batch_size, len(pairs))}/{len(pairs)}", flush=True)
        time.sleep(0.05)
    return out


def checkpoint_fetches(fetches: dict[str, dict], urls: set[str]) -> None:
    rows = [fetches[url] for url in sorted(urls) if url in fetches and fetches[url].get("ok")]
    write_jsonl(CANDIDATE_FETCH_CACHE, rows)


def merge_curation_fetch(fetches: dict[str, dict], url: str, new_fetch: dict) -> None:
    if not new_fetch.get("ok"):
        return
    old_fetch = fetches.get(url, {})
    new_len = int(new_fetch.get("text_len", 0) or 0)
    old_len = int(old_fetch.get("text_len", 0) or 0)
    if old_fetch.get("ok") and old_len >= 500 and new_len < 500:
        merged = dict(old_fetch)
        merged["api_text_for_curation"] = new_fetch.get("text", "")
        merged["api_title_for_curation"] = new_fetch.get("title", "")
        fetches[url] = merged
    else:
        fetches[url] = new_fetch


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\[[^\]]*\]", "", str(text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def clean_sentence(sentence: str) -> str:
    sentence = re.sub(r"\([^)]{0,120}\)", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence.strip(" .")


def answer_aliases(row: dict, fetch_audit: Any) -> list[str]:
    aliases = set(fetch_audit.answer_aliases(row.get("target_answer", "")))
    aliases.add(str(row.get("target_answer", "") or "").strip())
    aliases.add(str(row.get("target_evidence_title", "") or "").strip())
    aliases.add(wiki_title_from_url(row.get("target_evidence_url", "")))
    return [alias for alias in sorted(aliases, key=len, reverse=True) if alias]


def contains_answer(text: str, row: dict, fetch_audit: Any) -> bool:
    return fetch_audit.contains_any(fetch_audit.normalize_text(text), answer_aliases(row, fetch_audit))


def has_country_relation(sentence: str, row: dict) -> bool:
    country = row.get("country", "")
    sentence_norm = norm(sentence)
    answer_norm = norm(row.get("target_answer", ""))
    if any(marker in answer_norm for marker in COUNTRY_MARKERS.get(country, [])):
        return True
    for marker in COUNTRY_MARKERS.get(country, []):
        marker_re = re.escape(marker).replace("\\ ", r"\s+")
        if re.search(rf"\b{marker_re}\b", sentence_norm):
            if marker.endswith(("ian", "ese", "ish", "can", "i")) or marker in {"filipino", "british", "american"}:
                return True
            relation_patterns = [
                rf"\bin\s+{marker_re}\b",
                rf"\bof\s+{marker_re}\b",
                rf"\bfrom\s+{marker_re}\b",
                rf"\bfor\s+{marker_re}\b",
                rf"\bbased\s+in\s+{marker_re}\b",
                rf"\blocated\s+in\s+{marker_re}\b",
                rf"\b{marker_re}\s+(?:city|town|province|district|region|state|national|government|television|radio|company|bank|stadium|university|school|stock|exchange)\b",
            ]
            if any(re.search(pattern, sentence_norm) for pattern in relation_patterns):
                return True
    return False


def sentence_supports_row(sentence: str, row: dict, fetch_audit: Any) -> bool:
    sentence_norm = norm(sentence)
    if len(sentence) < 35 or len(sentence) > 320:
        return False
    if any(fragment in sentence_norm for fragment in BAD_TEXT_FRAGMENTS):
        return False
    return contains_answer(sentence, row, fetch_audit) and has_country_relation(sentence, row)


def answer_pattern(row: dict, fetch_audit: Any) -> str:
    return "|".join(re.escape(alias) for alias in answer_aliases(row, fetch_audit))


def predicate_after_answer(sentence: str, row: dict, fetch_audit: Any) -> str:
    cleaned = clean_sentence(sentence)
    pattern = answer_pattern(row, fetch_audit)
    if not pattern:
        return ""
    leading = re.match(
        rf"^(?:the\s+)?(?:{pattern})(?:\s*,\s*[^,]{{2,120}}\s*,)?\s+"
        rf"(?P<predicate>(?:is|was|are|were|has|had|serves|served|operates|operated|became|becomes|remains)\b.+)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if leading:
        return leading.group("predicate").strip(" .")
    known_as = re.search(
        rf"\b(?:known as|known professionally as|known by (?:his|her|their) stage name|popularly known as|better known as)\s+"
        rf"(?:{pattern})\s*,?\s+(?P<predicate>(?:is|was|are|were)\b.+)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if known_as:
        return known_as.group("predicate").strip(" .")
    return ""


def infer_entity_type(row: dict, predicate: str) -> str:
    answer_blob = f" {norm(row.get('target_answer', '') + ' ' + row.get('target_evidence_title', ''))} "
    pred_blob = f" {norm(predicate)} "
    answer_checks = [
        ("stock exchange", [" stock exchange "]),
        ("stadium", [" stadium ", " arena "]),
        ("airport", [" airport "]),
        ("port", [" port "]),
        ("university", [" university ", " college "]),
        ("school", [" school "]),
        ("government body", [" ministry ", " department ", " authority ", " commission ", " agency ", " council ", " office ", " board "]),
        ("company", [" company ", " corporation ", " bank ", " airline ", " group ", " plc ", " limited ", " ltd "]),
        ("newspaper", [" newspaper ", " daily "]),
        ("football club", [" football club ", " fc "]),
        ("festival", [" festival "]),
        ("examination", [" examination ", " certificate "]),
        ("river", [" river "]),
    ]
    for label, needles in answer_checks:
        if any(needle in answer_blob for needle in needles):
            return label
    if any(f" {word} " in pred_blob for word in PERSON_WORDS):
        return "person"
    predicate_checks = [
        ("television series", [" television series ", " animated television series "]),
        ("city", [" city ", " capital city "]),
        ("town", [" town ", " municipality "]),
        ("district", [" district "]),
        ("region", [" province ", " region ", " state "]),
        ("stadium", [" stadium ", " arena "]),
        ("airport", [" airport "]),
        ("port", [" port "]),
        ("university", [" university ", " college "]),
        ("school", [" school "]),
        ("government body", [" ministry ", " department ", " authority ", " commission ", " agency ", " council ", " office ", " board "]),
        ("company", [" company ", " corporation ", " bank ", " airline ", " media group ", " conglomerate ", " operator "]),
        ("newspaper", [" newspaper ", " daily "]),
        ("broadcaster", [" television network ", " broadcaster ", " radio station ", " tv channel "]),
        ("stock exchange", [" stock exchange "]),
        ("football club", [" football club "]),
        ("festival", [" festival "]),
        ("examination", [" examination ", " certificate "]),
        ("holiday", [" holiday ", " celebration "]),
        ("competition", [" competition ", " tournament ", " league ", " championship "]),
    ]
    for label, needles in predicate_checks:
        if any(needle in pred_blob for needle in needles):
            return label
    return ""


def choose_support_sentence(row: dict, fetches: dict[str, dict], fetch_audit: Any) -> tuple[str, str, str]:
    url = row.get("target_evidence_url", "")
    fetch = fetches.get(url, {})
    texts = [
        ("metadata_excerpt", row.get("target_evidence_excerpt", "")),
        ("api_curation_text", fetch.get("api_text_for_curation", "")),
        ("wikipedia_lead", fetch.get("text", "")),
    ]
    for source, text in texts:
        for sentence in split_sentences(text)[:5]:
            sentence = clean_sentence(sentence)
            if not sentence_supports_row(sentence, row, fetch_audit):
                continue
            predicate = predicate_after_answer(sentence, row, fetch_audit)
            if not predicate:
                continue
            if contains_answer(predicate, row, fetch_audit):
                continue
            if norm(predicate).startswith(("established ", "founded ", "located ")):
                continue
            entity_type = infer_entity_type(row, predicate)
            if not entity_type:
                continue
            return sentence, predicate, entity_type
    return "", "", ""


def make_question(row: dict, predicate: str, entity_type: str, fetch_audit: Any) -> str:
    predicate = predicate.strip(" .")
    if len(predicate) > 210:
        predicate = re.split(r",\s+which\b|,\s+and\b|;\s+", predicate)[0]
    if len(predicate) > 210:
        return ""
    question = f"In {row.get('country')}, which {entity_type} {predicate[0].lower() + predicate[1:]}?"
    question_norm = norm(question)
    if any(fragment in question_norm for fragment in BAD_QUESTION_FRAGMENTS):
        return ""
    if re.search(r"\b(?:and|or|of|in|the|a|an|with)\?$", question_norm):
        return ""
    if contains_answer(question, row, fetch_audit):
        return ""
    return question


def high_conf_issue(row: dict, fetches: dict[str, dict], builder: Any) -> bool:
    text = builder.evidence_cue_text(row, "target", fetches)
    return any(cue["norm"] not in text for cue in builder.high_confidence_question_cues(row))


def anchor_issue(row: dict, fetches: dict[str, dict], fetch_audit: Any) -> bool:
    fetch = fetches.get(row.get("target_evidence_url", ""), {})
    anchor_text = " ".join(
        str(part or "")
        for part in [
            row.get("target_evidence_title", ""),
            row.get("target_evidence_excerpt", ""),
            fetch.get("title", ""),
            str(fetch.get("text", ""))[:1500],
        ]
    )
    return not fetch_audit.contains_any(fetch_audit.normalize_text(anchor_text), answer_aliases(row, fetch_audit))


def structural_ok_for_curation(row: dict, fetch_audit: Any) -> bool:
    options = row.get("options") or []
    norm_options = [norm(option) for option in options]
    if len(options) != 4 or len(norm_options) != len(set(norm_options)):
        return False
    answer_norm = norm(row.get("target_answer", ""))
    if sum(1 for option in options if norm(option) == answer_norm) != 1:
        return False
    if norm(row.get("correct_answer", "")) != answer_norm:
        return False
    if not row.get("target_evidence_url") or not row.get("target_answer"):
        return False
    return bool(fetch_audit.answer_aliases(row.get("target_answer", "")))


def main() -> None:
    explicit_audit = load_module(AUDIT_EXPLICIT_SCRIPT, "audit48")
    fetch_audit = load_module(FETCH_AUDIT_SCRIPT, "fetch32")
    builder = load_module(BUILDER_SCRIPT, "builder45")

    fetches: dict[str, dict] = {}
    for cache in FETCH_CACHES:
        if cache == CANDIDATE_FETCH_CACHE:
            continue
        fetches.update(explicit_audit.load_fetch_cache(cache))
    for url, fetch in explicit_audit.load_fetch_cache(CANDIDATE_FETCH_CACHE).items():
        merge_curation_fetch(fetches, url, fetch)

    ambiguous_ids = {row["source_row_id"] for row in read_jsonl(AMBIGUOUS)}
    selected = read_jsonl(SOURCE_STRICT)
    used_sources = {row["source_row_id"] for row in selected}
    used_questions = {norm(row["question"]) for row in selected}
    counts = Counter(row["country"] for row in selected)

    pool_rows = []
    seen_pool = set()
    for pool in CANDIDATE_POOLS:
        for row in read_jsonl(pool):
            source_id = row.get("source_row_id", "")
            if source_id in seen_pool:
                continue
            seen_pool.add(source_id)
            if source_id in ambiguous_ids:
                continue
            if row.get("country") not in COUNTRY_ORDER:
                continue
            pool_rows.append(dict(row))

    strict_existing_by_country: dict[str, list[dict]] = defaultdict(list)
    curation_sources_by_country: dict[str, list[dict]] = defaultdict(list)
    initial_reject_counts = Counter()
    for row in pool_rows:
        country = row["country"]
        if row.get("source_row_id") in used_sources:
            continue
        failures, warnings = explicit_audit.audit_explicit_row(row, fetches, ambiguous_ids, fetch_audit)
        if not failures and not warnings and not high_conf_issue(row, fetches, builder) and not anchor_issue(row, fetches, fetch_audit):
            strict_existing_by_country[country].append(row)
            continue
        if structural_ok_for_curation(row, fetch_audit) and wiki_title_from_url(row.get("target_evidence_url", "")):
            curation_sources_by_country[country].append(row)
        else:
            initial_reject_counts["not_structurally_curatable"] += 1

    for country in COUNTRY_ORDER:
        strict_existing_by_country[country].sort(key=rank)
        curation_sources_by_country[country].sort(key=rank)

    added_existing = []
    for country in COUNTRY_ORDER:
        for row in strict_existing_by_country[country]:
            if counts[country] >= 1000:
                break
            question_key = norm(row.get("question", ""))
            source_id = row.get("source_row_id", "")
            if source_id in used_sources or question_key in used_questions:
                continue
            row = dict(row)
            row["split_name"] = "LocalNewsQA-Explicit-Strict-Defensible-1000-Curated"
            selected.append(row)
            used_sources.add(source_id)
            used_questions.add(question_key)
            counts[country] += 1
            added_existing.append(row)

    deficit_countries = [country for country in COUNTRY_ORDER if counts[country] < 1000]
    wiki_fetches: dict[str, dict] = {}
    seen_wiki_urls: set[str] = set()
    wikipedia_lead_urls_requested = 0

    curated_rows = []
    curation_log = []
    reject_log = []
    checkpoint_urls = {
        row.get("target_evidence_url", "")
        for country in deficit_countries
        for row in curation_sources_by_country[country]
        if row.get("target_evidence_url", "")
    }
    for country in deficit_countries:
        country_sources = curation_sources_by_country[country]
        for batch_start in range(0, len(country_sources), 80):
            if counts[country] >= 1000:
                break
            batch_sources = country_sources[batch_start : batch_start + 80]
            batch_urls = []
            for source in batch_sources:
                url = source.get("target_evidence_url", "")
                if url and url not in seen_wiki_urls and needs_wikipedia_api_hydration(url, fetches.get(url, {})):
                    seen_wiki_urls.add(url)
                    batch_urls.append(url)
            if batch_urls:
                wikipedia_lead_urls_requested += len(batch_urls)
                new_fetches = fetch_wikipedia_leads(batch_urls)
                wiki_fetches.update(new_fetches)
                for url, fetch in new_fetches.items():
                    merge_curation_fetch(fetches, url, fetch)
                checkpoint_fetches(fetches, checkpoint_urls)
                print(
                    f"{country}: counts={counts[country]}/1000 after hydrating batch {batch_start // 80 + 1}",
                    flush=True,
                )
            for source in batch_sources:
                if counts[country] >= 1000:
                    break
                if source.get("source_row_id") in used_sources:
                    continue
                original_question_key = norm(source.get("question", ""))
                if original_question_key not in used_questions:
                    original = dict(source)
                    original["split_name"] = "LocalNewsQA-Explicit-Strict-Defensible-1000-Curated"
                    original["curation_status"] = "curated_original_after_full_wikipedia_hydration"
                    failures, warnings = explicit_audit.audit_explicit_row(original, fetches, ambiguous_ids, fetch_audit)
                    if (
                        not failures
                        and not warnings
                        and not high_conf_issue(original, fetches, builder)
                        and not anchor_issue(original, fetches, fetch_audit)
                    ):
                        selected.append(original)
                        used_sources.add(original["source_row_id"])
                        used_questions.add(original_question_key)
                        counts[country] += 1
                        curated_rows.append(original)
                        curation_log.append(
                            {
                                "country": country,
                                "source_row_id": original.get("source_row_id", ""),
                                "old_question": source.get("question", ""),
                                "new_question": source.get("question", ""),
                                "target_answer": original.get("target_answer", ""),
                                "target_evidence_title": original.get("target_evidence_title", ""),
                                "target_evidence_url": original.get("target_evidence_url", ""),
                                "entity_type": "original",
                                "support_sentence": original.get("target_evidence_excerpt", ""),
                                "curation_status": original["curation_status"],
                            }
                        )
                        continue
                support_sentence, predicate, entity_type = choose_support_sentence(source, fetches, fetch_audit)
                if not support_sentence:
                    reject_log.append(
                        {
                            "country": country,
                            "source_row_id": source.get("source_row_id", ""),
                            "reason": "no_strict_support_sentence",
                            "question": source.get("question", ""),
                            "target_answer": source.get("target_answer", ""),
                            "target_evidence_title": source.get("target_evidence_title", ""),
                        }
                    )
                    continue
                new_question = make_question(source, predicate, entity_type, fetch_audit)
                if not new_question:
                    reject_log.append(
                        {
                            "country": country,
                            "source_row_id": source.get("source_row_id", ""),
                            "reason": "question_generation_reject",
                            "support_sentence": support_sentence,
                            "target_answer": source.get("target_answer", ""),
                            "target_evidence_title": source.get("target_evidence_title", ""),
                        }
                    )
                    continue
                question_key = norm(new_question)
                if question_key in used_questions:
                    reject_log.append(
                        {
                            "country": country,
                            "source_row_id": source.get("source_row_id", ""),
                            "reason": "duplicate_question",
                            "question": new_question,
                            "target_answer": source.get("target_answer", ""),
                        }
                    )
                    continue
                candidate = dict(source)
                candidate["question"] = new_question
                candidate["localized_question"] = new_question
                candidate["original_question"] = source.get("question", "")
                candidate["split_name"] = "LocalNewsQA-Explicit-Strict-Defensible-1000-Curated"
                candidate["split_type"] = "explicit"
                candidate["ambiguity_flag"] = False
                candidate["evidence_hint"] = support_sentence
                candidate["target_evidence_excerpt"] = support_sentence
                candidate["curation_status"] = "manual_rule_curated_from_supported_evidence"
                candidate["curation_support_sentence"] = support_sentence
                failures, warnings = explicit_audit.audit_explicit_row(candidate, fetches, ambiguous_ids, fetch_audit)
                if failures or warnings:
                    reject_log.append(
                        {
                            "country": country,
                            "source_row_id": source.get("source_row_id", ""),
                            "reason": "audit_reject",
                            "failures": " | ".join(failures),
                            "warnings": " | ".join(warnings),
                            "question": new_question,
                            "target_answer": source.get("target_answer", ""),
                        }
                    )
                    continue
                if high_conf_issue(candidate, fetches, builder) or anchor_issue(candidate, fetches, fetch_audit):
                    reject_log.append(
                        {
                            "country": country,
                            "source_row_id": source.get("source_row_id", ""),
                            "reason": "strict_probe_reject",
                            "question": new_question,
                            "target_answer": source.get("target_answer", ""),
                        }
                    )
                    continue
                selected.append(candidate)
                used_sources.add(candidate["source_row_id"])
                used_questions.add(question_key)
                counts[country] += 1
                curated_rows.append(candidate)
                curation_log.append(
                    {
                        "country": country,
                        "source_row_id": candidate.get("source_row_id", ""),
                        "old_question": source.get("question", ""),
                        "new_question": new_question,
                        "target_answer": candidate.get("target_answer", ""),
                        "target_evidence_title": candidate.get("target_evidence_title", ""),
                        "target_evidence_url": candidate.get("target_evidence_url", ""),
                        "entity_type": entity_type,
                        "support_sentence": support_sentence,
                        "curation_status": candidate["curation_status"],
                    }
                )
            print(f"{country}: counts={counts[country]}/1000 after curation batch", flush=True)

    selected.sort(key=lambda row: (COUNTRY_ORDER.index(row["country"]), rank(row)))
    for idx, row in enumerate(selected, start=1):
        row["id"] = f"localnewsqa_explicit_strict_defensible_1000_{idx:05d}"
        row["split_name"] = "LocalNewsQA-Explicit-Strict-Defensible-1000-Curated"

    source_ids = [row["source_row_id"] for row in selected]
    question_keys = [norm(row["question"]) for row in selected]
    selected_urls = {row.get("target_evidence_url", "") for row in selected if row.get("target_evidence_url", "")}
    validation_errors = []
    for country in COUNTRY_ORDER:
        if counts[country] != 1000:
            validation_errors.append(f"{country}: expected 1000, got {counts[country]}")
    if len(selected) != 17000:
        validation_errors.append(f"expected 17000 rows, got {len(selected)}")
    if len(source_ids) != len(set(source_ids)):
        validation_errors.append("duplicate source ids")
    if len(question_keys) != len(set(question_keys)):
        validation_errors.append("duplicate questions")
    if set(source_ids) & ambiguous_ids:
        validation_errors.append(f"ambiguous overlap: {len(set(source_ids) & ambiguous_ids)}")

    write_jsonl(OUT_JSONL, selected)
    write_csv(OUT_CSV, selected)
    write_csv(CURATION_LOG, curation_log)
    write_csv(REJECT_LOG, reject_log)
    write_jsonl(FETCH_CACHE, [fetches[url] for url in sorted(selected_urls) if url in fetches])
    checkpoint_fetches(fetches, checkpoint_urls)

    summary = {
        "rows": len(selected),
        "country_counts": dict(Counter(row["country"] for row in selected)),
        "source_split_type_counts": dict(Counter(row.get("source_split_type", "") for row in selected)),
        "added_existing_strict_rows": len(added_existing),
        "curated_rows": len(curated_rows),
        "curated_by_country": dict(Counter(row["country"] for row in curated_rows)),
        "candidate_existing_strict_counts": {country: len(strict_existing_by_country[country]) for country in COUNTRY_ORDER},
        "candidate_curation_source_counts": {country: len(curation_sources_by_country[country]) for country in COUNTRY_ORDER},
        "initial_reject_counts": dict(initial_reject_counts),
        "curation_reject_counts": dict(Counter(row.get("reason", "") for row in reject_log)),
        "wikipedia_lead_urls_requested": wikipedia_lead_urls_requested,
        "wikipedia_lead_extracts_loaded": sum(1 for fetch in wiki_fetches.values() if fetch.get("ok")),
        "duplicate_source_ids": len(source_ids) - len(set(source_ids)),
        "duplicate_questions": len(question_keys) - len(set(question_keys)),
        "ambiguous_overlap": len(set(source_ids) & ambiguous_ids),
        "valid": not validation_errors,
        "validation_errors": validation_errors,
        "paths": {
            "jsonl": str(OUT_JSONL),
            "csv": str(OUT_CSV),
            "fetch_cache": str(FETCH_CACHE),
            "candidate_fetch_cache": str(CANDIDATE_FETCH_CACHE),
            "curation_log": str(CURATION_LOG),
            "reject_log": str(REJECT_LOG),
            "summary": str(SUMMARY),
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if validation_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
