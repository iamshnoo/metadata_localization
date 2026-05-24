#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


SOURCE_JSONL = Path(
    "./qa_data/localnewsqa_core/runs/core_20260408_v3_nonbatch/generation_candidates_pruned_v1_normalized.jsonl"
)

OVERLAY_CSVS = [
    Path("./qa_data/localnewsqa_core/runs/human_validation_ambiguous_510_web.csv"),
    Path("./qa_data/localnewsqa_core/runs/human_validation_ambiguous_374_web_certified_complete.csv"),
]

FIELDNAMES = [
    "id",
    "country",
    "continent",
    "topic",
    "year",
    "question",
    "options",
    "target_country",
    "contrast_country",
    "target_answer",
    "contrast_answer",
    "evidence_hint",
    "target_query",
    "target_evidence_url",
    "target_evidence_title",
    "target_evidence_snippet",
    "target_evidence_excerpt",
    "target_match_type",
    "contrast_query",
    "contrast_evidence_url",
    "contrast_evidence_title",
    "contrast_evidence_snippet",
    "contrast_evidence_excerpt",
    "contrast_match_type",
    "judge_target_factuality",
    "judge_locale_dependence",
    "judge_no_explicit_leakage",
    "annotator_notes",
]


def signature(row):
    return (
        str(row.get("country", "")).strip(),
        str(row.get("topic", "")).strip(),
        str(row.get("year", "")).strip(),
        str(row.get("question", "")).strip(),
        str(row.get("target_answer", "")).strip(),
        str(row.get("contrast_answer", "")).strip(),
        str(row.get("contrast_country", "")).strip(),
    )


def load_overlays():
    merged = {}
    for path in OVERLAY_CSVS:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                merged[signature(row)] = row
    return merged


def main():
    parser = argparse.ArgumentParser(description="Build a full ambiguous validation pool with prior evidence overlays.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    overlays = load_overlays()
    rows = []
    with SOURCE_JSONL.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            if not item.get("ambiguity_flag"):
                continue
            row = {
                "id": f"localnewsqa_ambig_{idx:07d}",
                "country": item["country"],
                "continent": item["continent"],
                "topic": item["topic"],
                "year": item["year"],
                "question": item["question"],
                "options": " || ".join(item["options"]),
                "target_country": item["country"],
                "contrast_country": item["contrast_country"],
                "target_answer": item["target_answer"],
                "contrast_answer": item["contrast_answer"],
                "evidence_hint": item.get("evidence_hint", ""),
                "target_query": "",
                "target_evidence_url": "",
                "target_evidence_title": "",
                "target_evidence_snippet": "",
                "target_evidence_excerpt": "",
                "target_match_type": "",
                "contrast_query": "",
                "contrast_evidence_url": "",
                "contrast_evidence_title": "",
                "contrast_evidence_snippet": "",
                "contrast_evidence_excerpt": "",
                "contrast_match_type": "",
                "judge_target_factuality": "",
                "judge_locale_dependence": "",
                "judge_no_explicit_leakage": "",
                "annotator_notes": "",
            }
            over = overlays.get(signature(row))
            if over:
                for field in FIELDNAMES:
                    if field == "id":
                        continue
                    if over.get(field, ""):
                        row[field] = over[field]
            rows.append(row)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "rows": len(rows),
        "countries": dict(Counter(r["country"] for r in rows)),
        "topics": dict(Counter(r["topic"] for r in rows)),
        "target_evidence_rows": sum(bool(str(r["target_evidence_url"]).strip()) for r in rows),
        "contrast_evidence_rows": sum(bool(str(r["contrast_evidence_url"]).strip()) for r in rows),
        "both_evidence_rows": sum(
            bool(str(r["target_evidence_url"]).strip()) and bool(str(r["contrast_evidence_url"]).strip()) for r in rows
        ),
        "overlay_matches": sum(1 for r in rows if r["target_evidence_url"] or r["contrast_evidence_url"]),
        "output_csv": str(out_path),
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
