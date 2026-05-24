#!/usr/bin/env python3

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


DEFAULT_CANDIDATE_JSONL = (
    "./qa_data/localnewsqa_core/runs/"
    "core_20260408_v3_nonbatch/generation_candidates_pruned_v1_normalized.jsonl"
)


def format_options(options):
    return " || ".join(f"{chr(65 + idx)}: {opt}" for idx, opt in enumerate(options))


def candidate_to_row(row, fieldnames):
    out = {field: "" for field in fieldnames}
    out.update(
        {
            "id": row.get("generation_custom_id", ""),
            "country": row.get("country", ""),
            "continent": row.get("continent", ""),
            "topic": row.get("topic", ""),
            "year": row.get("year", ""),
            "question": row.get("question", ""),
            "options": format_options(row.get("options", [])),
            "target_country": row.get("target_country", ""),
            "contrast_country": row.get("contrast_country", ""),
            "target_answer": row.get("target_answer", ""),
            "contrast_answer": row.get("contrast_answer", ""),
            "evidence_hint": row.get("evidence_hint", ""),
        }
    )
    return out


def load_seed_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames, rows


def load_ambiguous_candidates(path):
    by_country = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("split_type", "")).lower() != "ambiguous":
                continue
            by_country[str(row.get("country", "")).strip()].append(row)
    return by_country


def main():
    parser = argparse.ArgumentParser(
        description="Expand a LocalNewsQA ambiguous human-validation sheet while preserving existing prefills."
    )
    parser.add_argument("--seed-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--per-country", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-jsonl", default=DEFAULT_CANDIDATE_JSONL)
    args = parser.parse_args()

    fieldnames, seed_rows = load_seed_rows(args.seed_csv)
    candidates_by_country = load_ambiguous_candidates(args.candidate_jsonl)

    seed_by_country = defaultdict(list)
    seen_ids = set()
    for row in seed_rows:
        country = str(row.get("country", "")).strip()
        row_id = str(row.get("id", "")).strip()
        seed_by_country[country].append(row)
        if row_id:
            seen_ids.add(row_id)

    rng = random.Random(args.seed)
    output_rows = []
    summary = {}

    for country in sorted(candidates_by_country):
        current = list(seed_by_country.get(country, []))
        if len(current) > args.per_country:
            raise SystemExit(
                f"Seed CSV already has {len(current)} rows for {country}; cannot shrink to {args.per_country}."
            )

        pool = [
            row
            for row in candidates_by_country[country]
            if str(row.get("generation_custom_id", "")).strip() not in seen_ids
        ]
        rng.shuffle(pool)

        needed = args.per_country - len(current)
        if len(pool) < needed:
            raise SystemExit(
                f"Country '{country}' has only {len(pool)} remaining candidates after excluding seeded rows; "
                f"need {needed}."
            )

        additions = [candidate_to_row(row, fieldnames) for row in pool[:needed]]
        merged = current + additions
        output_rows.extend(merged)
        summary[country] = {
            "seed_rows": len(current),
            "added_rows": len(additions),
            "total_rows": len(merged),
        }

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    report = {
        "seed_csv": args.seed_csv,
        "candidate_jsonl": args.candidate_jsonl,
        "per_country": args.per_country,
        "seed": args.seed,
        "total_rows": len(output_rows),
        "countries": summary,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
