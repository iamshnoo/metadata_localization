#!/usr/bin/env python3

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


GENERIC_TERMS = {
    "board",
    "cabinet",
    "chairman",
    "commission",
    "committee",
    "court",
    "council",
    "day",
    "department",
    "federation",
    "general",
    "government",
    "governor",
    "holiday",
    "minister",
    "ministry",
    "office",
    "president",
    "prime",
    "radio",
    "speaker",
    "tribunal",
}


def stable_score(seed: int, text: str) -> int:
    digest = hashlib.sha1(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def answer_quality(answer: str) -> int:
    toks = [t for t in str(answer or "").lower().replace("-", " ").split() if t]
    if not toks:
        return 0
    generic_penalty = sum(t in GENERIC_TERMS for t in toks)
    return len(" ".join(toks)) + 4 * len(toks) - 6 * generic_penalty


def evidence_count(row: dict) -> int:
    return int(bool(str(row.get("target_evidence_url", "")).strip())) + int(
        bool(str(row.get("contrast_evidence_url", "")).strip())
    )


def main():
    parser = argparse.ArgumentParser(description="Build reserve replacement candidates for unresolved LocalNewsQA validation rows.")
    parser.add_argument("--sample-csv", required=True)
    parser.add_argument("--pool-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--multiplier", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sample_rows = list(csv.DictReader(open(args.sample_csv, newline="", encoding="utf-8")))
    pool_rows = list(csv.DictReader(open(args.pool_csv, newline="", encoding="utf-8")))
    fieldnames = list(pool_rows[0].keys())

    unresolved = [
        r
        for r in sample_rows
        if str(r.get("target_source_certified", "")).strip().lower() != "yes"
        or str(r.get("contrast_source_certified", "")).strip().lower() != "yes"
    ]
    need = Counter((r["country"], r["topic"]) for r in unresolved)

    used_questions = {r["question"] for r in sample_rows}
    by_bucket = defaultdict(list)
    for row in pool_rows:
        key = (row["country"], row["topic"])
        if key not in need:
            continue
        if row["question"] in used_questions:
            continue
        by_bucket[key].append(row)

    selected = []
    summary = {"unresolved_rows": len(unresolved), "buckets": {}}
    for key, count in sorted(need.items()):
        candidates = by_bucket[key]
        candidates = sorted(
            candidates,
            key=lambda r: (
                -evidence_count(r),
                -(answer_quality(r.get("target_answer", "")) + answer_quality(r.get("contrast_answer", ""))),
                stable_score(args.seed, r["id"]),
            ),
        )
        take = min(len(candidates), count * args.multiplier)
        chosen = candidates[:take]
        selected.extend(chosen)
        summary["buckets"][f"{key[0]} | {key[1]}"] = {
            "need": count,
            "available": len(candidates),
            "selected": len(chosen),
        }

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    import json

    summary["rows"] = len(selected)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
