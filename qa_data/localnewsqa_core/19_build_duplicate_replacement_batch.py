#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from random import Random


def evidence_score(row):
    return int(bool(str(row.get("target_evidence_url", "")).strip())) + int(
        bool(str(row.get("contrast_evidence_url", "")).strip())
    )


def duplicate_replacement_rows(sample_rows):
    by_question = defaultdict(list)
    for idx, row in enumerate(sample_rows):
        by_question[row["question"].strip()].append((idx, row))

    duplicates = []
    for question, items in by_question.items():
        if len(items) <= 1:
            continue
        items = sorted(items, key=lambda pair: pair[1]["id"])
        for _, row in items[1:]:
            duplicates.append(row)
    return duplicates


def main():
    parser = argparse.ArgumentParser(description="Build same-bucket reserve candidates for duplicate-question replacement.")
    parser.add_argument("--sample-csv", required=True)
    parser.add_argument("--pool-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--multiplier", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sample_rows = list(csv.DictReader(open(args.sample_csv, newline="", encoding="utf-8")))
    fieldnames = list(sample_rows[0].keys())
    pool_rows = list(csv.DictReader(open(args.pool_csv, newline="", encoding="utf-8")))

    duplicate_rows = duplicate_replacement_rows(sample_rows)
    needed = Counter((row["country"], row["topic"]) for row in duplicate_rows)
    used_questions = {row["question"].strip() for row in sample_rows}
    used_ids = {row["id"] for row in sample_rows}

    rng = Random(args.seed)
    selected = []
    bucket_summary = {}
    for key in sorted(needed):
        country, topic = key
        candidates = [
            dict(row)
            for row in pool_rows
            if row["country"] == country
            and row["topic"] == topic
            and row["id"] not in used_ids
            and row["question"].strip() not in used_questions
        ]
        rng.shuffle(candidates)
        candidates.sort(
            key=lambda row: (
                -evidence_score(row),
                row["question"].strip(),
                row["id"],
            )
        )
        take = min(len(candidates), needed[key] * args.multiplier)
        chosen = candidates[:take]
        selected.extend(chosen)
        bucket_summary[f"{country} | {topic}"] = {
            "need": needed[key],
            "available": len(candidates),
            "selected": len(chosen),
        }

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    summary = {
        "duplicate_rows": len(duplicate_rows),
        "rows": len(selected),
        "bucket_summary": bucket_summary,
        "output_csv": str(out_path),
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
