#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGETQA = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/localnewsqa_targetqa_explicit_style.jsonl"
)
DEFAULT_AMBIGUOUS = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_semantic_gold_ambiguous_1700/localnewsqa_ambiguous_semantic_gold_1700.jsonl"
)
DEFAULT_OUTDIR = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_semantic_gold_ambiguous_1700"
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def csv_value(value: Any) -> str:
    if isinstance(value, list):
        return " || ".join(str(part) for part in value)
    return "" if value is None else str(value)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(value) for key, value in row.items()})


def question_with_country(country: str, question: str) -> str:
    question = str(question or "").strip()
    display_country = {
        "United States": "the United States",
        "United Kingdom": "the United Kingdom",
        "Philippines": "the Philippines",
    }.get(country, country)
    if not question:
        return f"In {display_country},"
    if country and country.lower() in question.lower():
        return question
    first = question[0].lower() + question[1:] if len(question) > 1 else question.lower()
    return f"In {display_country}, {first}"


def localize_row(row: dict) -> dict:
    out = dict(row)
    localized = question_with_country(out.get("country", ""), out.get("question", ""))
    out["question"] = localized
    out["localized_question"] = localized
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter TargetQA explicit-style split to be disjoint from final ambiguous split.")
    parser.add_argument("--targetqa", type=Path, default=DEFAULT_TARGETQA)
    parser.add_argument("--ambiguous", type=Path, default=DEFAULT_AMBIGUOUS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    target_rows = read_jsonl(args.targetqa)
    ambiguous_source_ids = {row["source_row_id"] for row in read_jsonl(args.ambiguous)}
    kept = [localize_row(row) for row in target_rows if row.get("source_row_id") not in ambiguous_source_ids]
    removed = [row for row in target_rows if row.get("source_row_id") in ambiguous_source_ids]

    jsonl_path = args.outdir / "localnewsqa_targetqa_explicit_style_disjoint_from_web_ambiguous.jsonl"
    csv_path = args.outdir / "localnewsqa_targetqa_explicit_style_disjoint_from_web_ambiguous.csv"
    removed_path = args.outdir / "targetqa_rows_removed_due_to_web_ambiguous_overlap.csv"
    write_jsonl(jsonl_path, kept)
    write_csv(csv_path, kept)
    write_csv(removed_path, removed)

    summary = {
        "targetqa_input": str(args.targetqa),
        "ambiguous_input": str(args.ambiguous),
        "targetqa_input_rows": len(target_rows),
        "ambiguous_rows": len(ambiguous_source_ids),
        "targetqa_output_rows": len(kept),
        "removed_overlap_rows": len(removed),
        "output_source_split_type_counts": dict(Counter(row.get("source_split_type", "") for row in kept)),
        "removed_source_split_type_counts": dict(Counter(row.get("source_split_type", "") for row in removed)),
        "paths": {
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
            "removed_overlap_rows": str(removed_path),
            "summary": str(args.outdir / "targetqa_disjoint_summary.json"),
        },
    }
    (args.outdir / "targetqa_disjoint_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
