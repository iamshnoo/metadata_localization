#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_semantic_gold_ambiguous_1700"
)
AUDIT_SCRIPT = ROOT / "qa_data/localnewsqa_core/32_web_audit_ambiguous_verifiable.py"
WEAK_SCRIPT = ROOT / "qa_data/localnewsqa_core/26_flag_weak_locale_ambiguous.py"

GENERIC_CUES = {
    "A",
    "An",
    "BBC",
    "Business",
    "Can",
    "Company",
    "Court",
    "Department",
    "Exchange",
    "Finance",
    "Government",
    "House",
    "Inc",
    "International",
    "Minister",
    "Ministry",
    "National",
    "News",
    "Office",
    "Official",
    "Parliament",
    "President",
    "Prime Minister",
    "Radio",
    "Secretary",
    "Senate",
    "State",
    "TV",
    "Television",
    "The",
    "University",
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


def contains_phrase(text: str, phrase: str) -> bool:
    phrase_norm = norm(phrase)
    return bool(phrase_norm) and f" {phrase_norm} " in f" {norm(text)} "


def answer_contains_cue(row: dict, cue: str, audit_mod: Any) -> bool:
    cue_norm = norm(cue)
    if not cue_norm:
        return False
    aliases = set()
    aliases.update(audit_mod.answer_aliases(row.get("target_answer", "")))
    aliases.update(audit_mod.answer_aliases(row.get("contrast_answer", "")))
    return any(cue_norm == alias or f" {cue_norm} " in f" {alias} " or f" {alias} " in f" {cue_norm} " for alias in aliases)


def evidence_text(row: dict, side: str, fetches: dict[str, dict]) -> str:
    url = row.get(f"{side}_evidence_url", "")
    fetch = fetches.get(url, {})
    return " ".join(
        [
            row.get(f"{side}_evidence_title", ""),
            row.get(f"{side}_evidence_snippet", ""),
            row.get(f"{side}_evidence_excerpt", ""),
            fetch.get("title", ""),
            fetch.get("text", ""),
        ]
    )


def cue_candidates(question: str, weak_mod: Any) -> list[str]:
    cues = []
    cues.extend(match.strip() for match in re.findall(r"[\"']([^\"']{2,80})[\"']", question))
    cues.extend(weak_mod.extract_named_spans(question))
    for token in re.findall(r"\b[A-Z0-9]{2,}(?:[-/][A-Z0-9]{2,})*\b", question):
        cues.append(token)
    cleaned = []
    for cue in cues:
        cue = re.sub(r"\s+", " ", str(cue or "")).strip(" ,.;:!?()[]{}")
        if not cue:
            continue
        cue_norm = norm(cue)
        if len(cue_norm) < 3 or cue in GENERIC_CUES or cue_norm in {norm(item) for item in GENERIC_CUES}:
            continue
        if cue_norm.isdigit():
            continue
        cleaned.append(cue)
    return list(dict.fromkeys(cleaned))


def reviewer_risks(row: dict, fetches: dict[str, dict], audit_mod: Any, weak_mod: Any) -> list[dict]:
    target_text = evidence_text(row, "target", fetches)
    contrast_text = evidence_text(row, "contrast", fetches)
    risks = []
    for cue in cue_candidates(row.get("question", ""), weak_mod):
        if answer_contains_cue(row, cue, audit_mod):
            continue
        target_hit = contains_phrase(target_text, cue)
        contrast_hit = contains_phrase(contrast_text, cue)
        if target_hit == contrast_hit:
            continue
        risks.append(
            {
                "cue": cue,
                "risk": "target_only_visible_cue" if target_hit else "contrast_only_visible_cue",
                "target_hit": "true" if target_hit else "false",
                "contrast_hit": "true" if contrast_hit else "false",
            }
        )
    return risks


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit subjective reviewer-risk cues in ambiguous questions.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--input", type=Path, default=DEFAULT_DIR / "localnewsqa_ambiguous_semantic_gold_1700.jsonl")
    parser.add_argument("--fetches", type=Path, default=DEFAULT_DIR / "semantic_gold_selected_evidence_fetches.jsonl")
    args = parser.parse_args()

    audit_mod = load_module(AUDIT_SCRIPT, "audit32")
    weak_mod = load_module(WEAK_SCRIPT, "weak26")

    rows = read_jsonl(args.input)
    fetches = {row["url"]: row for row in read_jsonl(args.fetches)}

    risk_rows = []
    risk_counts_by_country = Counter()
    cue_counts = Counter()
    for row in rows:
        risks = reviewer_risks(row, fetches, audit_mod, weak_mod)
        if risks:
            risk_counts_by_country[row.get("country", "")] += 1
        for risk in risks:
            cue_counts[risk["cue"]] += 1
            risk_rows.append(
                {
                    "source_row_id": row.get("source_row_id", ""),
                    "country": row.get("country", ""),
                    "contrast_country": row.get("contrast_country", ""),
                    "topic": row.get("topic", ""),
                    "question": row.get("question", ""),
                    "target_answer": row.get("target_answer", ""),
                    "contrast_answer": row.get("contrast_answer", ""),
                    "cue": risk["cue"],
                    "risk": risk["risk"],
                    "target_evidence_title": row.get("target_evidence_title", ""),
                    "contrast_evidence_title": row.get("contrast_evidence_title", ""),
                }
            )

    by_country_rows = [
        {"country": country, "risk_row_count": count}
        for country, count in sorted(risk_counts_by_country.items())
    ]
    summary = {
        "input": str(args.input),
        "rows": len(rows),
        "risk_row_count": len({row["source_row_id"] for row in risk_rows}),
        "risk_cue_count": len(risk_rows),
        "risk_counts_by_country": dict(risk_counts_by_country),
        "top_cues": dict(cue_counts.most_common(50)),
        "paths": {
            "risk_rows": str(args.outdir / "semantic_gold_reviewer_risk_rows.csv"),
            "risk_counts_by_country": str(args.outdir / "semantic_gold_reviewer_risk_by_country.csv"),
            "summary": str(args.outdir / "semantic_gold_reviewer_risk_summary.json"),
        },
    }

    write_csv(args.outdir / "semantic_gold_reviewer_risk_rows.csv", risk_rows)
    write_csv(args.outdir / "semantic_gold_reviewer_risk_by_country.csv", by_country_rows)
    (args.outdir / "semantic_gold_reviewer_risk_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
