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
BASE = ROOT / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_semantic_gold_ambiguous_1700"
DEFAULT_INPUT = BASE / "explicit_max_audit/strict_1000/localnewsqa_targetqa_explicit_strict_1000_per_country_polished.jsonl"
DEFAULT_OUTDIR = BASE / "explicit_max_audit/strict_1000"
FETCH_CACHES = [
    BASE / "explicit_max_audit/explicit_target_evidence_fetches.jsonl",
    BASE / "explicit_max_audit/strict_1000/polished_audit/explicit_target_evidence_fetches.jsonl",
]
AUDIT_SCRIPT = ROOT / "qa_data/localnewsqa_core/32_web_audit_ambiguous_verifiable.py"

STOPWORDS = {
    "about",
    "after",
    "also",
    "annual",
    "before",
    "being",
    "between",
    "country",
    "during",
    "from",
    "government",
    "local",
    "major",
    "market",
    "mentioned",
    "national",
    "news",
    "often",
    "public",
    "reports",
    "story",
    "their",
    "this",
    "which",
    "with",
    "would",
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


def load_fetch_cache(path: Path) -> dict[str, dict]:
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


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def norm(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_any_alias(text_norm: str, aliases: list[str], audit_mod: Any) -> bool:
    return audit_mod.contains_any(text_norm, aliases)


def evidence_texts(row: dict, fetches: dict[str, dict]) -> tuple[str, str]:
    fetch = fetches.get(row.get("target_evidence_url", ""), {})
    lead = " ".join(
        str(part or "")
        for part in [
            row.get("target_evidence_title", ""),
            row.get("target_evidence_excerpt", ""),
            fetch.get("title", ""),
            str(fetch.get("text", ""))[:1500],
        ]
    )
    full = " ".join(
        str(part or "")
        for part in [
            row.get("target_evidence_url", ""),
            row.get("target_evidence_title", ""),
            row.get("target_evidence_excerpt", ""),
            fetch.get("title", ""),
            fetch.get("text", ""),
        ]
    )
    return audit_mod_norm(lead), audit_mod_norm(full)


def audit_mod_norm(text: Any) -> str:
    return norm(text)


def question_cues(row: dict) -> list[str]:
    q = norm(row.get("question", ""))
    answer = set(norm(row.get("target_answer", "")).split())
    country = set(norm(row.get("country", "")).split())
    cues = []
    for token in q.split():
        if len(token) < 4 or token in STOPWORDS or token in answer or token in country:
            continue
        if token.isdigit():
            continue
        cues.append(token)
    return sorted(set(cues))


def year_tokens(question: str) -> list[str]:
    return sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", question or "")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe explicit rows for answer-anchor evidence issues.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    audit_mod = load_module(AUDIT_SCRIPT, "audit32")
    fetches = {}
    for cache in FETCH_CACHES:
        fetches.update(load_fetch_cache(cache))

    rows = read_jsonl(args.input)
    issue_rows = []
    answer_not_tied_rows = []
    severity_counts = Counter()
    issue_counts = Counter()
    for row in rows:
        lead_norm, full_norm = evidence_texts(row, fetches)
        issues = []
        severity = "pass"
        aliases = audit_mod.answer_aliases(row.get("target_answer", ""))
        if not contains_any_alias(lead_norm, aliases, audit_mod):
            issues.append("answer_not_tied_to_evidence_title_or_lead")
            severity = "fail"

        years = year_tokens(row.get("question", ""))
        missing_years = [year for year in years if year not in full_norm]
        if missing_years:
            issues.append("diagnostic_temporal_year_not_supported:" + ",".join(missing_years))
            if severity == "pass":
                severity = "warn"

        cues = question_cues(row)
        cue_hits = [cue for cue in cues if cue in full_norm]
        if len(cues) >= 4 and len(cue_hits) < 2:
            issues.append("diagnostic_low_question_relation_cue_overlap")
            if severity == "pass":
                severity = "warn"

        severity_counts[severity] += 1
        for issue in issues:
            issue_counts[issue.split(":", 1)[0]] += 1
        if issues:
            out = {
                "source_row_id": row.get("source_row_id", ""),
                "country": row.get("country", ""),
                "severity": severity,
                "issues": " | ".join(issues),
                "question": row.get("question", ""),
                "target_answer": row.get("target_answer", ""),
                "target_evidence_title": row.get("target_evidence_title", ""),
                "target_evidence_url": row.get("target_evidence_url", ""),
                "cue_count": len(cues),
                "cue_hits": " | ".join(cue_hits[:20]),
            }
            issue_rows.append(out)
            if "answer_not_tied_to_evidence_title_or_lead" in issues:
                answer_not_tied_rows.append(out)

    issue_path = args.outdir / "strict_explicit_answer_anchor_probe_nonpass.csv"
    answer_path = args.outdir / "strict_explicit_answer_anchor_probe_answer_not_tied.csv"
    summary_path = args.outdir / "strict_explicit_answer_anchor_probe_summary.json"
    write_csv(issue_path, issue_rows)
    write_csv(answer_path, answer_not_tied_rows)
    summary = {
        "rows": len(rows),
        "nonpass": len(issue_rows),
        "answer_not_tied_rows": len(answer_not_tied_rows),
        "severity_counts": dict(severity_counts),
        "issue_counts": dict(issue_counts),
        "paths": {
            "nonpass": str(issue_path),
            "answer_not_tied": str(answer_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
