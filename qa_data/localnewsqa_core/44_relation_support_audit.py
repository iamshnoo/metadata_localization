#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_semantic_gold_ambiguous_1700"
)
AUDIT_SCRIPT = ROOT / "qa_data/localnewsqa_core/32_web_audit_ambiguous_verifiable.py"
REPAIR_SCRIPT = ROOT / "qa_data/localnewsqa_core/43_repair_semantic_gold_questions.py"

STOPWORDS = {
    "a",
    "about",
    "after",
    "again",
    "all",
    "also",
    "an",
    "and",
    "annual",
    "answer",
    "answers",
    "area",
    "as",
    "at",
    "be",
    "became",
    "been",
    "before",
    "being",
    "best",
    "body",
    "called",
    "commonly",
    "context",
    "country",
    "described",
    "during",
    "event",
    "fit",
    "fits",
    "free",
    "for",
    "from",
    "had",
    "has",
    "have",
    "held",
    "in",
    "is",
    "it",
    "its",
    "local",
    "main",
    "major",
    "most",
    "name",
    "national",
    "news",
    "of",
    "office",
    "often",
    "on",
    "or",
    "public",
    "reference",
    "referenced",
    "relevant",
    "report",
    "reporting",
    "reports",
    "role",
    "said",
    "serves",
    "serving",
    "special",
    "that",
    "the",
    "their",
    "through",
    "to",
    "used",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "would",
    "year",
    "wikipedia",
}

ROLE_CUES = {
    "leader": ["leader", "head", "government", "president", "prime minister", "office"],
    "prime minister": ["prime minister", "head of government"],
    "president": ["president", "head of state"],
    "foreign": ["foreign", "affairs", "secretary", "minister", "diplomat"],
    "defence": ["defence", "defense", "armed forces", "military", "minister", "secretary"],
    "health": ["health", "minister", "department"],
    "transport": ["transport", "transportation", "minister", "department"],
    "finance": ["finance", "budget", "treasury", "fiscal", "minister", "chancellor"],
    "education": ["education", "school", "university", "exam", "student"],
    "tax": ["tax", "revenue", "customs"],
    "broadcast": ["broadcast", "radio", "television", "media"],
    "legislature": ["legislature", "parliament", "congress", "chamber", "house", "senate"],
    "court": ["court", "judicial", "justice"],
    "airport": ["airport", "international", "gateway"],
    "city": ["city", "town", "capital"],
    "market": ["market", "exchange", "stock", "index", "business"],
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
        writer.writerows(rows)


def norm(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def padded(text: str) -> str:
    return f" {norm(text)} "


def contains_phrase(text: str, phrase: str) -> bool:
    phrase_norm = norm(phrase)
    return bool(phrase_norm) and f" {phrase_norm} " in padded(text)


def tokens(text: str) -> list[str]:
    return [tok for tok in norm(text).split() if len(tok) > 2 and tok not in STOPWORDS]


def answer_tokens(answer: str) -> set[str]:
    return set(tokens(answer)) | {tok for tok in norm(answer).split() if len(tok) > 1}


def side_text(row: dict, side: str, fetches: dict[str, dict]) -> str:
    url = row.get(f"{side}_evidence_url", "")
    fetch = fetches.get(url, {})
    return " ".join(
        [
            row.get(f"{side}_evidence_title", ""),
            row.get(f"{side}_evidence_excerpt", ""),
            fetch.get("title", ""),
            fetch.get("text", ""),
        ]
    )


def title_text(row: dict, side: str, fetches: dict[str, dict]) -> str:
    url = row.get(f"{side}_evidence_url", "")
    fetch = fetches.get(url, {})
    return " ".join([row.get(f"{side}_evidence_title", ""), fetch.get("title", ""), url])


def title_supports_answer(answer: str, title_blob: str, evidence: str) -> bool:
    aliases = {answer, answer.replace("&", "and"), answer.replace(".", "")}
    aliases.add(re.sub(r"^(the|a|an)\s+", "", answer, flags=re.IGNORECASE).strip())
    title_norm = padded(title_blob)
    evidence_head = padded(evidence[:4000])
    for alias in aliases:
        alias_norm = norm(alias)
        if alias_norm and f" {alias_norm} " in title_norm:
            return True
        if alias_norm and f" {alias_norm} " in evidence_head:
            return True
    ans_toks = {tok for tok in answer_tokens(answer) if len(tok) > 2}
    title_toks = set(tokens(title_blob))
    if ans_toks and ans_toks <= title_toks:
        return True
    if len(ans_toks) >= 2 and len(ans_toks & title_toks) >= max(1, len(ans_toks) - 1):
        return True
    return False


def answer_supported_near_relation(answer: str, evidence: str, cues: set[str], window: int = 260) -> bool:
    evidence_norm = norm(evidence[:40000])
    aliases = {norm(answer), norm(answer.replace("&", "and")), norm(answer.replace(".", ""))}
    aliases = {alias for alias in aliases if alias}
    for alias in aliases:
        for match in re.finditer(rf"\b{re.escape(alias)}\b", evidence_norm):
            start = max(0, match.start() - window)
            end = min(len(evidence_norm), match.end() + window)
            nearby = evidence_norm[start:end]
            if any(f" {cue} " in f" {nearby} " for cue in cues):
                return True
    return False


def year_supported(year: str, evidence: str) -> bool:
    if not re.fullmatch(r"(?:19|20)\d{2}", str(year or "")):
        return True
    y = int(year)
    if str(y) in evidence:
        return True
    for start, end in re.findall(r"\b((?:19|20)\d{2})\s*(?:-|to|and|/|\u2013|\u2014)\s*((?:19|20)\d{2})\b", evidence):
        s, e = int(start), int(end)
        if s <= y <= e:
            return True
    for start in re.findall(r"\bfrom\s+((?:19|20)\d{2})\b", evidence, flags=re.IGNORECASE):
        if int(start) <= y:
            return True
    return False


def person_like(answer: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z'.-]*", str(answer or ""))
    if len(words) < 2:
        return False
    non_person = {
        "airport",
        "assembly",
        "authority",
        "bank",
        "board",
        "bureau",
        "commission",
        "company",
        "council",
        "court",
        "department",
        "exchange",
        "festival",
        "governor",
        "group",
        "house",
        "justice",
        "minister",
        "ministry",
        "office",
        "party",
        "president",
        "secretary",
        "school",
        "senate",
        "service",
        "university",
    }
    return not (set(tokens(answer)) & non_person)


def needs_temporal_support(row: dict, side: str) -> bool:
    answer = row.get(f"{side}_answer", "")
    question = str(row.get("question", ""))
    if not person_like(answer):
        return False
    return bool(
        re.search(
            r"\b(relevant year|in \d{4}|during \d{4}|after \d{4}|before \d{4}|served|held|became|appointed|minister|leader|president|prime minister)\b",
            question,
            flags=re.IGNORECASE,
        )
    )


def relation_cues(row: dict) -> set[str]:
    # Relation support must be visible from the public QA item itself.  The
    # internal evidence_hint can contain target-specific scaffolding, so using
    # it here would let a weak question pass on hidden metadata.
    text = f"{row.get('question', '')} {row.get('topic', '')}"
    cues = set(tokens(text))
    answer_side_tokens = answer_tokens(row.get("target_answer", "")) | answer_tokens(row.get("contrast_answer", ""))
    country_tokens = set(tokens(row.get("country", ""))) | set(tokens(row.get("contrast_country", "")))
    cues -= answer_side_tokens
    cues -= country_tokens
    expanded = set(cues)
    text_norm = norm(text)
    for key, vals in ROLE_CUES.items():
        if key in text_norm or any(val in text_norm for val in vals):
            expanded.update(tok for val in vals for tok in tokens(val))
    return {cue for cue in expanded if cue not in STOPWORDS and len(cue) > 2}


def cue_hits(cues: set[str], evidence: str) -> set[str]:
    evidence_norm = padded(evidence[:30000])
    return {cue for cue in cues if f" {cue} " in evidence_norm}


def first_sentence(evidence: str) -> str:
    text = re.sub(r"\s+", " ", evidence).strip()
    # Strip common navigation blocks from scraped Wikipedia pages.
    marker = "from wikipedia"
    lower = text.lower()
    if marker in lower:
        text = text[lower.index(marker) :]
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:3])[:1200]


def side_relation_audit(row: dict, side: str, fetches: dict[str, dict], audit_mod: Any) -> dict:
    answer = row.get(f"{side}_answer", "")
    evidence = side_text(row, side, fetches)
    title_blob = title_text(row, side, fetches)
    base = audit_mod.audit_side(row, side, fetches)
    cues = relation_cues(row)
    hits = cue_hits(cues, evidence)
    first = first_sentence(evidence)
    first_hits = cue_hits(cues, first)
    answer_title = title_supports_answer(answer, title_blob, evidence)
    answer_relation_nearby = answer_supported_near_relation(answer, evidence, cues)
    temporal_question = needs_temporal_support(row, side)
    year_ok = year_supported(row.get("year", ""), evidence) if temporal_question else True

    issues = []
    if not base.get(f"{side}_url_ok"):
        issues.append("url_not_fetchable")
    if not base.get(f"{side}_answer_found"):
        issues.append("answer_not_found")
    if not base.get(f"{side}_country_marker_found"):
        issues.append("country_marker_not_found")
    if not answer_title and not answer_relation_nearby:
        issues.append("answer_not_tied_to_evidence_title_or_lead")
    if temporal_question and not year_ok:
        issues.append("temporal_year_not_supported")

    # We treat title/lead answer support plus country support as the hard floor. Cue hits
    # distinguish full relation support from merely an entity page mention.
    answer_anchor_ok = answer_title or answer_relation_nearby
    strong_relation = answer_anchor_ok and bool(first_hits) and year_ok
    acceptable_relation = answer_anchor_ok and year_ok and (len(hits) >= 2 or bool(first_hits) or answer_relation_nearby)
    if not acceptable_relation:
        issues.append("low_question_relation_cue_overlap")

    severity = "pass"
    if any(issue in issues for issue in ["url_not_fetchable", "answer_not_found", "country_marker_not_found", "answer_not_tied_to_evidence_title_or_lead", "temporal_year_not_supported"]):
        severity = "fail"
    elif issues:
        severity = "warn"
    return {
        "source_row_id": row["source_row_id"],
        "country": row["country"],
        "contrast_country": row.get("contrast_country", ""),
        "side": side,
        "answer": answer,
        "question": row.get("question", ""),
        "year": row.get("year", ""),
        "evidence_title": row.get(f"{side}_evidence_title", ""),
        "evidence_url": row.get(f"{side}_evidence_url", ""),
        "severity": severity,
        "issues": " | ".join(issues),
        "answer_title_or_lead_support": answer_title,
        "answer_near_relation_support": answer_relation_nearby,
        "year_support": year_ok,
        "cue_count": len(cues),
        "cue_hits": " | ".join(sorted(hits)),
        "first_sentence_cue_hits": " | ".join(sorted(first_hits)),
        "strong_relation": strong_relation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether each evidence page supports the question-answer relation.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--input", type=Path, default=DEFAULT_DIR / "localnewsqa_ambiguous_semantic_gold_1700.jsonl")
    parser.add_argument("--fetches", type=Path, default=DEFAULT_DIR / "semantic_gold_selected_evidence_fetches.jsonl")
    args = parser.parse_args()

    audit_mod = load_module(AUDIT_SCRIPT, "audit32")
    rows = read_jsonl(args.input)
    fetches = {row["url"]: row for row in read_jsonl(args.fetches)}
    side_audits = []
    for row in rows:
        side_audits.append(side_relation_audit(row, "target", fetches, audit_mod))
        side_audits.append(side_relation_audit(row, "contrast", fetches, audit_mod))

    failing = [row for row in side_audits if row["severity"] == "fail"]
    warning = [row for row in side_audits if row["severity"] == "warn"]
    summary = {
        "input": str(args.input),
        "fetches": str(args.fetches),
        "rows": len(rows),
        "side_audits": len(side_audits),
        "side_severity_counts": dict(Counter(row["severity"] for row in side_audits)),
        "issue_counts": dict(Counter(issue for row in side_audits for issue in row["issues"].split(" | ") if issue)),
        "strong_relation_side_count": sum(1 for row in side_audits if row["strong_relation"]),
        "valid_no_failures": not failing,
        "warning_sides": len(warning),
        "failing_sides": len(failing),
    }
    out_path = args.outdir / "semantic_gold_relation_support_audit.csv"
    fail_path = args.outdir / "semantic_gold_relation_support_failures.csv"
    warn_path = args.outdir / "semantic_gold_relation_support_warnings.csv"
    summary_path = args.outdir / "semantic_gold_relation_support_summary.json"
    write_csv(out_path, side_audits)
    write_csv(fail_path, failing)
    write_csv(warn_path, warning)
    summary["paths"] = {
        "audit": str(out_path),
        "failures": str(fail_path),
        "warnings": str(warn_path),
        "summary": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
