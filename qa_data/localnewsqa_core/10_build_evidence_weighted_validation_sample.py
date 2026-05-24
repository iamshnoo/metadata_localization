#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def stable_score(seed: int, row_id: str) -> int:
    digest = hashlib.sha1(f"{seed}:{row_id}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def bool_int(value: str) -> int:
    return 1 if str(value).strip() else 0


def evidence_score(row: dict) -> tuple:
    target = bool_int(row.get("target_evidence_url", ""))
    contrast = bool_int(row.get("contrast_evidence_url", ""))
    any_url = 1 if (target or contrast) else 0
    factual = 1 if row.get("judge_target_factuality", "") == "yes" else 0
    locale = 1 if row.get("judge_locale_dependence", "") == "yes" else 0
    return (
        target + contrast,
        any_url,
        factual + locale,
    )


def sort_rows(rows, seed: int):
    return sorted(
        rows,
        key=lambda r: (
            -evidence_score(r)[0],
            -evidence_score(r)[1],
            -evidence_score(r)[2],
            stable_score(seed, r.get("id", r.get("question", ""))),
        ),
    )


def allocate_topic_quotas(topic_counts: Counter, target_total: int) -> dict:
    topics = [topic for topic, count in sorted(topic_counts.items()) if count > 0]
    if len(topics) > target_total:
        raise ValueError(f"More topics ({len(topics)}) than target slots ({target_total})")

    quotas = {topic: 1 for topic in topics}
    remaining_slots = target_total - len(topics)
    if remaining_slots <= 0:
        return quotas

    remaining_counts = {topic: topic_counts[topic] - 1 for topic in topics}
    remaining_total = sum(remaining_counts.values())
    if remaining_total <= 0:
        return quotas

    fractional = {}
    for topic in topics:
        exact = remaining_slots * (remaining_counts[topic] / remaining_total)
        add = int(exact)
        quotas[topic] += add
        fractional[topic] = exact - add

    used = sum(quotas.values())
    leftovers = target_total - used
    for topic, _ in sorted(fractional.items(), key=lambda kv: (-kv[1], kv[0])):
        if leftovers <= 0:
            break
        quotas[topic] += 1
        leftovers -= 1

    return quotas


def main():
    parser = argparse.ArgumentParser(
        description="Build a country/topic-balanced LocalNewsQA human-validation sheet from the web-evidence pool."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--per-country", type=int, default=22)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    by_country = defaultdict(list)
    for row in rows:
        by_country[row["country"]].append(row)

    sampled = []
    summary = {
        "input_csv": str(input_path),
        "per_country": args.per_country,
        "seed": args.seed,
        "countries": {},
    }

    for country in sorted(by_country):
        country_rows = by_country[country]
        if len(country_rows) < args.per_country:
            raise SystemExit(
                f"Country '{country}' has only {len(country_rows)} rows; need {args.per_country}."
            )

        topic_counts = Counter(row["topic"] for row in country_rows)
        quotas = allocate_topic_quotas(topic_counts, args.per_country)

        chosen = []
        chosen_ids = set()
        for topic in sorted(quotas):
            topic_rows = [row for row in country_rows if row["topic"] == topic]
            ranked = sort_rows(topic_rows, args.seed)
            take = min(quotas[topic], len(ranked))
            for row in ranked[:take]:
                chosen.append(row)
                chosen_ids.add(row["id"])

        if len(chosen) < args.per_country:
            remainder = [row for row in country_rows if row["id"] not in chosen_ids]
            ranked_remainder = sort_rows(remainder, args.seed)
            needed = args.per_country - len(chosen)
            chosen.extend(ranked_remainder[:needed])

        chosen = sort_rows(chosen, args.seed)
        sampled.extend(chosen)

        target_urls = sum(bool_int(row.get("target_evidence_url", "")) for row in chosen)
        contrast_urls = sum(bool_int(row.get("contrast_evidence_url", "")) for row in chosen)
        either_urls = sum(
            1
            for row in chosen
            if row.get("target_evidence_url", "").strip() or row.get("contrast_evidence_url", "").strip()
        )
        summary["countries"][country] = {
            "selected_rows": len(chosen),
            "topic_counts": dict(Counter(row["topic"] for row in chosen)),
            "target_evidence_rows": target_urls,
            "contrast_evidence_rows": contrast_urls,
            "either_evidence_rows": either_urls,
        }

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled)

    summary["total_rows"] = len(sampled)
    summary["target_evidence_rows"] = sum(
        bool_int(row.get("target_evidence_url", "")) for row in sampled
    )
    summary["contrast_evidence_rows"] = sum(
        bool_int(row.get("contrast_evidence_url", "")) for row in sampled
    )
    summary["either_evidence_rows"] = sum(
        1
        for row in sampled
        if row.get("target_evidence_url", "").strip() or row.get("contrast_evidence_url", "").strip()
    )

    summary_path = (
        Path(args.summary_json)
        if args.summary_json
        else output_path.with_name(output_path.stem + "_summary.json")
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote sample to {output_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    raise SystemExit(main())
