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
    pattern = r"(.+?)_eval_(target|contrast|auto)_(with_metadata|without_metadata)_(custom|llama3_chat)_(.+?)(?:_c(\d+))?\.jsonl$"
    m = re.search(pattern, path.name)
    if not m:
        return {}
    dataset_name = m.group(1)
    locale_role = m.group(2)
    metadata_flag = m.group(3) == "with_metadata"
    model_type = m.group(4)
    base_slug = m.group(5)
    corruption_pct = m.group(6)
    corruption_rate = float(corruption_pct) / 100.0 if corruption_pct is not None else 0.0
    base_url = slug_to_url.get(base_slug, base_slug)
    return {
        "dataset_name": dataset_name,
        "locale_role": locale_role,
        "model_type": model_type,
        "metadata": metadata_flag,
        "base_url": base_url,
        "base_slug": base_slug,
        "url_corruption_rate": corruption_rate,
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
        row.get("eval_correct_answer", row.get("correct_answer", "")),
        row.get("target_answer", ""),
        row.get("contrast_answer", ""),
        row.get("country", ""),
        row.get("target_country", ""),
        row.get("contrast_country", ""),
        row.get("continent", ""),
        row.get("generated_by", ""),
        row.get("split_family", ""),
        row.get("split_type", ""),
        str(row.get("year", "")),
        row.get("topic", ""),
    ]
    blob = "||".join(parts).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def outcome_from_row(row: dict) -> str:
    processed = row.get("processed_answer")
    if processed is None:
        return "skipped"
    is_correct = row.get("is_correct")
    if is_correct is None:
        is_correct = processed == row.get("eval_correct_answer", row.get("correct_answer"))
    return "correct" if is_correct else "incorrect"


def outcome_from_flags(correct: int, incorrect: int, skipped: int) -> str:
    if correct == 1:
        return "correct"
    if incorrect in (-1, 1):
        return "incorrect"
    if skipped == 1:
        return "skipped"
    return "skipped"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build question-level outcome matrix across models and URLs."
    )
    parser.add_argument(
        "--results-dir",
        default="/path/to/metacul/results/downstream",
        help="Directory containing *_eval_*.jsonl files.",
    )
    parser.add_argument(
        "--output-csv",
        default="/path/to/metacul/results/downstream_eval_matrix.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--only-metadata",
        action="store_true",
        default=False,
        help="Only include metadata=True variants in the output matrix.",
    )
    parser.add_argument(
        "--c0-parity-from",
        default=None,
        help="Optional CSV from a baseline run to override corruption=0 outcomes.",
    )
    args = parser.parse_args()

    slug_to_url = {slugify_url(u): u for u in URLS}
    results_dir = Path(args.results_dir)
    output_csv = Path(args.output_csv)

    rows_by_key = {}
    if args.only_metadata:
        variants = [
            "llama3_chat_with_metadata",
            "custom_with_metadata",
        ]
    else:
        variants = [
            "llama3_chat_with_metadata",
            "llama3_chat_without_metadata",
            "custom_with_metadata",
            "custom_without_metadata",
        ]

    for path in sorted(results_dir.glob("*_eval_*.jsonl")):
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
                row.get("eval_correct_answer"),
                row.get("split_type"),
                row.get("topic"),
                row.get("year"),
                row.get("country"),
                row.get("target_country"),
                row.get("contrast_country"),
                row.get("continent"),
                row.get("generated_by"),
                info["url_corruption_rate"],
                info["locale_role"],
            )
            entry = rows_by_key.get(key)
            if entry is None:
                entry = {
                    "question_id": question_id,
                    "dataset_name": info["dataset_name"],
                    "locale_role": info["locale_role"],
                    "split_type": row.get("split_type"),
                    "topic": row.get("topic"),
                    "year": row.get("year"),
                    "country": row.get("country"),
                    "target_country": row.get("target_country"),
                    "contrast_country": row.get("contrast_country"),
                    "continent": row.get("continent"),
                    "generated_by": row.get("generated_by"),
                    "base_url": info["base_url"],
                    "url_corruption_rate": info["url_corruption_rate"],
                    "outcomes": {},
                }
                rows_by_key[key] = entry
            entry["outcomes"][variant] = outcome_from_row(row)

    if args.c0_parity_from:
        baseline = Path(args.c0_parity_from)
        if baseline.exists():
            base_df = Path(args.c0_parity_from)
            import pandas as pd  # local import to avoid adding dependency if unused

            base = pd.read_csv(base_df)
            available = set(base.columns)
            variants_for_override = [v for v in variants if f"{v}_correct" in available]

            index = {}
            for entry in rows_by_key.values():
                key = (
                    entry["question_id"],
                    entry["base_url"],
                    entry["generated_by"],
                    entry["country"],
                    entry["continent"],
                    float(entry["url_corruption_rate"]),
                    entry["locale_role"],
                )
                index[key] = entry

            replaced = 0
            for _, row in base.iterrows():
                key = (
                    row.get("question_id"),
                    row.get("base_url"),
                    row.get("generated_by"),
                    row.get("country"),
                    row.get("continent"),
                    0.0,
                    row.get("locale_role", "target"),
                )
                entry = index.get(key)
                if entry is None:
                    continue
                for variant in variants_for_override:
                    outcome = outcome_from_flags(
                        int(row.get(f"{variant}_correct", 0)),
                        int(row.get(f"{variant}_incorrect", 0)),
                        int(row.get(f"{variant}_skipped", 0)),
                    )
                    entry["outcomes"][variant] = outcome
                replaced += 1
            print(f"[✔] Applied c0 parity overrides to {replaced} rows from {base_df}")
        else:
            print(f"[!] c0 parity file not found: {baseline}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "question_id",
            "dataset_name",
            "locale_role",
            "split_type",
            "topic",
            "year",
            "country",
            "target_country",
            "contrast_country",
            "continent",
            "generated_by",
            "base_url",
            "url_corruption_rate",
            "answered_by_all",
        ]
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
                entry["dataset_name"],
                entry["locale_role"],
                entry["split_type"],
                entry["topic"],
                entry["year"],
                entry["country"],
                entry["target_country"],
                entry["contrast_country"],
                entry["continent"],
                entry["generated_by"],
                entry["base_url"],
                f"{entry['url_corruption_rate']:.2f}",
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
            row.insert(6, 1 if answered_by_all else 0)
            writer.writerow(row)

    print(f"[✔] Wrote matrix to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
