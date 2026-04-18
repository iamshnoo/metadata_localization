#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_key(row):
    return (
        row.get("question"),
        tuple(row.get("options") or []),
        row.get("country"),
        row.get("target_country"),
        row.get("contrast_country"),
        row.get("split_type"),
        row.get("topic"),
        row.get("year"),
    )


def accuracy(rows):
    valid = [r for r in rows if r.get("processed_answer") is not None]
    if not valid:
        return 0.0, 0, 0
    correct = sum(1 for r in valid if r.get("is_correct"))
    return float(correct) / float(len(valid)), correct, len(valid)


def main():
    parser = argparse.ArgumentParser(description="Compute LocalNewsQA explicit/ambiguous and locale-sensitivity metrics.")
    parser.add_argument("--target-jsonl", required=True)
    parser.add_argument("--contrast-jsonl", default=None)
    args = parser.parse_args()

    target_rows = load_jsonl(args.target_jsonl)
    contrast_rows = load_jsonl(args.contrast_jsonl) if args.contrast_jsonl else []

    explicit_target = [r for r in target_rows if str(r.get("split_type", "")).lower() == "explicit"]
    ambiguous_target = [r for r in target_rows if str(r.get("split_type", "")).lower() == "ambiguous"]
    ambiguous_contrast = [r for r in contrast_rows if str(r.get("split_type", "")).lower() == "ambiguous"]

    explicit_acc, explicit_correct, explicit_total = accuracy(explicit_target)
    amb_target_acc, amb_target_correct, amb_target_total = accuracy(ambiguous_target)
    amb_contrast_acc, amb_contrast_correct, amb_contrast_total = accuracy(ambiguous_contrast)

    paired = []
    if contrast_rows:
        contrast_map = {row_key(r): r for r in ambiguous_contrast}
        for target in ambiguous_target:
            other = contrast_map.get(row_key(target))
            if other is not None:
                paired.append((target, other))

    locale_sensitive_n = 0
    locale_sensitive_correct = 0
    dominant_bias_n = 0
    dominant_bias_hits = 0
    for target, contrast in paired:
        target_answer = target.get("processed_answer")
        contrast_answer = contrast.get("processed_answer")
        if target_answer is None or contrast_answer is None:
            continue
        locale_sensitive_n += 1
        if target.get("is_correct") and contrast.get("is_correct") and target_answer != contrast_answer:
            locale_sensitive_correct += 1
        dominant_bias_n += 1
        if target_answer == contrast_answer and (
            bool(target.get("is_correct")) != bool(contrast.get("is_correct"))
        ):
            dominant_bias_hits += 1

    summary = {
        "explicit_accuracy": explicit_acc,
        "explicit_correct": explicit_correct,
        "explicit_total": explicit_total,
        "ambiguous_target_accuracy": amb_target_acc,
        "ambiguous_target_correct": amb_target_correct,
        "ambiguous_target_total": amb_target_total,
        "ambiguous_contrast_accuracy": amb_contrast_acc,
        "ambiguous_contrast_correct": amb_contrast_correct,
        "ambiguous_contrast_total": amb_contrast_total,
        "locale_sensitivity_accuracy": (
            float(locale_sensitive_correct) / float(locale_sensitive_n) if locale_sensitive_n else 0.0
        ),
        "locale_sensitivity_total": locale_sensitive_n,
        "dominant_locale_bias_rate": (
            float(dominant_bias_hits) / float(dominant_bias_n) if dominant_bias_n else 0.0
        ),
        "dominant_locale_bias_total": dominant_bias_n,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
