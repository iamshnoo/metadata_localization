#!/usr/bin/env python3
import argparse
import csv
import json
import re
import hashlib
from pathlib import Path


URLS = [
    "www.factquizmaster.com",
    "www.globalfactcheck.org",
    "www.worldknowledgehub.com",
    "www.civicspedia.org",
    "www.internationalfacts.net",
    "www.currentaffairsdesk.com",
    "www.newsinsightarchive.com",
    "www.globalquizvault.com",
    "www.factualdigest.org",
    "www.publicknowledgebase.net",
]


def slugify_url(url: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", url).strip("_")


def parse_results_filename(path: Path, slug_to_url: dict) -> dict:
    pattern = r"qa_metacul_eval_(with_metadata|without_metadata)_(custom|llama3_chat)_(.+)\.jsonl$"
    m = re.search(pattern, path.name)
    if not m:
        return {}
    metadata_flag = m.group(1) == "with_metadata"
    model_type = m.group(2)
    base_slug = m.group(3)
    base_url = slug_to_url.get(base_slug, base_slug)
    return {
        "model_type": model_type,
        "metadata": metadata_flag,
        "base_url": base_url,
        "base_slug": base_slug,
    }


def model_variant_name(model_type: str, metadata: bool) -> str:
    suffix = "with_metadata" if metadata else "without_metadata"
    return f"{model_type}_{suffix}"


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def compute_question_id(row: dict) -> str:
    parts = [
        row.get("question", ""),
        json.dumps(row.get("options", []), ensure_ascii=False),
        row.get("correct_answer", ""),
        row.get("country", ""),
        row.get("continent", ""),
        row.get("generated_by", ""),
    ]
    blob = "||".join(parts).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def outcome_from_row(row: dict) -> str:
    processed = row.get("processed_answer")
    if processed is None:
        return "skipped"
    is_correct = row.get("is_correct")
    if is_correct is None:
        is_correct = processed == row.get("correct_answer")
    return "correct" if is_correct else "incorrect"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build question-level outcome matrix across models and URLs."
    )
    parser.add_argument(
        "--results-dir",
        default="/scratch/$USER$/metacul/results/downstream",
        help="Directory containing qa_metacul_eval_*.jsonl files.",
    )
    parser.add_argument(
        "--output-csv",
        default="/scratch/$USER$/metacul/results/qa_metacul_eval.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    slug_to_url = {slugify_url(u): u for u in URLS}
    results_dir = Path(args.results_dir)
    output_csv = Path(args.output_csv)

    rows_by_key = {}
    variants = [
        "llama3_chat_with_metadata",
        "llama3_chat_without_metadata",
        "custom_with_metadata",
        "custom_without_metadata",
    ]

    for path in sorted(results_dir.glob("qa_metacul_eval_*.jsonl")):
        info = parse_results_filename(path, slug_to_url)
        if not info:
            continue
        variant = model_variant_name(info["model_type"], info["metadata"])
        for row in load_jsonl(path):
            question_id = compute_question_id(row)
            key = (
                info["base_url"],
                row.get("question"),
                tuple(row.get("options") or []),
                row.get("correct_answer"),
                row.get("country"),
                row.get("continent"),
                row.get("generated_by"),
            )
            entry = rows_by_key.get(key)
            if entry is None:
                entry = {
                    "question_id": question_id,
                    "country": row.get("country"),
                    "continent": row.get("continent"),
                    "generated_by": row.get("generated_by"),
                    "base_url": info["base_url"],
                    "outcomes": {},
                }
                rows_by_key[key] = entry
            entry["outcomes"][variant] = outcome_from_row(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["question_id", "country", "continent", "generated_by", "base_url", "answered_by_all"]
        for variant in variants:
            header.extend(
                [
                    f"{variant}_correct",
                    f"{variant}_incorrect",
                    f"{variant}_skipped",
                ]
            )
        writer.writerow(header)

        for entry in rows_by_key.values():
            row = [
                entry["question_id"],
                entry["country"],
                entry["continent"],
                entry["generated_by"],
                entry["base_url"],
            ]
            outcomes = entry["outcomes"]
            answered_by_all = True
            for variant in variants:
                outcome = outcomes.get(variant)
                if outcome == "correct":
                    row.extend([1, 0, 0])
                elif outcome == "incorrect":
                    row.extend([0, -1, 0])
                else:
                    answered_by_all = False
                    row.extend([0, 0, 0])
            row.insert(5, 1 if answered_by_all else 0)
            writer.writerow(row)

    print(f"[✔] Wrote matrix to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
