#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Replace unresolved validation rows with certified reserve candidates.")
    parser.add_argument("--sample-csv", required=True)
    parser.add_argument("--candidate-certified-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    sample_rows = list(csv.DictReader(open(args.sample_csv, newline="", encoding="utf-8")))
    fieldnames = list(sample_rows[0].keys())
    candidate_rows = list(csv.DictReader(open(args.candidate_certified_csv, newline="", encoding="utf-8")))

    certified_candidates = [
        r
        for r in candidate_rows
        if str(r.get("target_source_certified", "")).strip().lower() == "yes"
        and str(r.get("contrast_source_certified", "")).strip().lower() == "yes"
    ]
    by_bucket = defaultdict(list)
    for row in certified_candidates:
        by_bucket[(row["country"], row["topic"])].append(row)

    used_candidate_ids = set()
    replaced = 0
    unresolved_after = 0
    replacement_counts = Counter()
    final_rows = []

    for row in sample_rows:
        needs_replacement = (
            str(row.get("target_source_certified", "")).strip().lower() != "yes"
            or str(row.get("contrast_source_certified", "")).strip().lower() != "yes"
        )
        if not needs_replacement:
            final_rows.append(row)
            continue

        key = (row["country"], row["topic"])
        replacement = None
        for cand in by_bucket.get(key, []):
            if cand["id"] not in used_candidate_ids:
                replacement = dict(cand)
                break

        if replacement is None:
            unresolved_after += 1
            final_rows.append(row)
            continue

        used_candidate_ids.add(replacement["id"])
        replacement["annotator_notes"] = (
            str(replacement.get("annotator_notes", "")).strip()
            + (" | " if str(replacement.get("annotator_notes", "")).strip() else "")
            + f"replaced unresolved row {row['id']} from same country/topic"
        )
        final_rows.append(replacement)
        replaced += 1
        replacement_counts[key] += 1

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    summary = {
        "rows": len(final_rows),
        "replaced_rows": replaced,
        "remaining_unresolved": unresolved_after,
        "target_certified": sum(r.get("target_source_certified") == "yes" for r in final_rows),
        "contrast_certified": sum(r.get("contrast_source_certified") == "yes" for r in final_rows),
        "both_certified": sum(
            r.get("target_source_certified") == "yes" and r.get("contrast_source_certified") == "yes"
            for r in final_rows
        ),
        "judge_locale_dependence_yes": sum(r.get("judge_locale_dependence") == "yes" for r in final_rows),
        "replacement_counts": {f"{k[0]} | {k[1]}": v for k, v in sorted(replacement_counts.items())},
        "output_csv": str(out_path),
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
