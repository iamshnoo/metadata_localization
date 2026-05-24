#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def find_duplicate_rows(sample_rows):
    by_question = defaultdict(list)
    for idx, row in enumerate(sample_rows):
        by_question[row["question"].strip()].append((idx, row))

    replace_ids = set()
    for question, items in by_question.items():
        if len(items) <= 1:
            continue
        items = sorted(items, key=lambda pair: pair[1]["id"])
        for _, row in items[1:]:
            replace_ids.add(row["id"])
    return replace_ids


def main():
    parser = argparse.ArgumentParser(description="Replace duplicate-question rows with unique certified same-bucket candidates.")
    parser.add_argument("--sample-csv", required=True)
    parser.add_argument("--candidate-certified-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    sample_rows = list(csv.DictReader(open(args.sample_csv, newline="", encoding="utf-8")))
    fieldnames = list(sample_rows[0].keys())
    candidate_rows = list(csv.DictReader(open(args.candidate_certified_csv, newline="", encoding="utf-8")))

    replace_ids = find_duplicate_rows(sample_rows)
    used_ids = {row["id"] for row in sample_rows}
    used_questions = {row["question"].strip() for row in sample_rows if row["id"] not in replace_ids}

    by_bucket = defaultdict(list)
    for row in candidate_rows:
        if str(row.get("target_source_certified", "")).strip().lower() != "yes":
            continue
        if str(row.get("contrast_source_certified", "")).strip().lower() != "yes":
            continue
        by_bucket[(row["country"], row["topic"])].append(dict(row))

    replaced = 0
    unresolved = 0
    replacement_counts = Counter()
    duplicate_kept = 0
    final_rows = []

    for row in sample_rows:
        if row["id"] not in replace_ids:
            final_rows.append(row)
            continue

        key = (row["country"], row["topic"])
        replacement = None
        for cand in by_bucket.get(key, []):
            question = cand["question"].strip()
            if cand["id"] in used_ids or question in used_questions:
                continue
            replacement = cand
            break

        if replacement is None:
            unresolved += 1
            final_rows.append(row)
            used_questions.add(row["question"].strip())
            duplicate_kept += 1
            continue

        used_ids.add(replacement["id"])
        used_questions.add(replacement["question"].strip())
        replacement["annotator_notes"] = (
            str(replacement.get("annotator_notes", "")).strip()
            + (" | " if str(replacement.get("annotator_notes", "")).strip() else "")
            + f"replaced duplicate-template row {row['id']} from same country/topic"
        )
        final_rows.append(replacement)
        replaced += 1
        replacement_counts[key] += 1

    question_counts = Counter(row["question"].strip() for row in final_rows)
    duplicate_strings = {q: n for q, n in question_counts.items() if n > 1}

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    summary = {
        "rows": len(final_rows),
        "duplicate_rows_replaced": replaced,
        "replacement_shortfall": unresolved,
        "duplicate_rows_kept": duplicate_kept,
        "duplicate_question_strings": len(duplicate_strings),
        "duplicate_extra_rows": sum(n - 1 for n in duplicate_strings.values()),
        "target_certified": sum(row.get("target_source_certified") == "yes" for row in final_rows),
        "contrast_certified": sum(row.get("contrast_source_certified") == "yes" for row in final_rows),
        "both_certified": sum(
            row.get("target_source_certified") == "yes" and row.get("contrast_source_certified") == "yes"
            for row in final_rows
        ),
        "judge_locale_dependence_yes": sum(row.get("judge_locale_dependence") == "yes" for row in final_rows),
        "replacement_counts": {f"{country} | {topic}": n for (country, topic), n in sorted(replacement_counts.items())},
        "output_csv": str(out_path),
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
