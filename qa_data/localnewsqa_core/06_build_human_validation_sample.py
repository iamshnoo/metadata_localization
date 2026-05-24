#!/usr/bin/env python3

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


def format_options(options):
    return " || ".join(f"{chr(65 + idx)}: {opt}" for idx, opt in enumerate(options))


def main():
    parser = argparse.ArgumentParser(
        description="Build a stratified LocalNewsQA human-validation sample focused on ambiguous items."
    )
    parser.add_argument(
        "--dataset",
        default="iamshnoo/LocalNewsQA",
        help="HF dataset name or local datasets-compatible path",
    )
    parser.add_argument("--split", default="train", help="Dataset split to sample from")
    parser.add_argument(
        "--per-country",
        type=int,
        default=20,
        help="Number of ambiguous items to sample per country",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    ambiguous = [row for row in ds if str(row.get("split_type", "")).lower() == "ambiguous"]

    by_country = defaultdict(list)
    for row in ambiguous:
        by_country[str(row["country"]).strip()].append(dict(row))

    rng = random.Random(args.seed)
    sampled = []
    for country in sorted(by_country):
        rows = by_country[country]
        if len(rows) < args.per_country:
            raise SystemExit(
                f"Country '{country}' has only {len(rows)} ambiguous rows; "
                f"cannot sample {args.per_country}."
            )
        rng.shuffle(rows)
        chosen = rows[: args.per_country]
        for row in chosen:
            sampled.append(
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
                    "judge_target_factuality": "",
                    "judge_locale_dependence": "",
                    "judge_no_explicit_leakage": "",
                    "annotator_notes": "",
                }
            )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
        "judge_target_factuality",
        "judge_locale_dependence",
        "judge_no_explicit_leakage",
        "annotator_notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled)

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "per_country": args.per_country,
        "seed": args.seed,
        "total_rows": len(sampled),
        "countries": sorted(by_country.keys()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote annotation sheet to {output_path}")


if __name__ == "__main__":
    raise SystemExit(main())
