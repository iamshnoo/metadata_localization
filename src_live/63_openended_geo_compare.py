#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


CODE_TO_COUNTRY_NAME = {
    "us": "United States",
    "ca": "Canada",
    "jm": "Jamaica",
    "in": "India",
    "pk": "Pakistan",
    "bd": "Bangladesh",
    "lk": "Sri Lanka",
    "hk": "Hong Kong",
    "my": "Malaysia",
    "ph": "Philippines",
    "ng": "Nigeria",
    "za": "South Africa",
    "ke": "Kenya",
    "gh": "Ghana",
    "tz": "Tanzania",
    "gb": "United Kingdom",
    "ie": "Ireland",
    "cn": "China",
    "ir": "Iran",
    "gr": "Greece",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare open-ended CROQ-style runs with shared target-country mapping."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="JSONL output files to compare.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Where to write the combined comparison CSV.",
    )
    return parser.parse_args()


def normalized_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    probs = [count / total for count in counter.values() if count > 0]
    if len(probs) <= 1:
        return 0.0
    entropy = -sum(p * math.log(p) for p in probs)
    return float(entropy / math.log(len(probs)))


def read_rows(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def target_name_from_row(row: dict) -> Optional[str]:
    name = row.get("target_country_name") or row.get("country")
    if name:
        return str(name).strip() or None
    code = row.get("target_country_tag") or row.get("url_country_tag")
    if code:
        return CODE_TO_COUNTRY_NAME.get(str(code).strip().lower())
    return None


def build_target_lookup(files: List[Path]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for path in files:
        for row in read_rows(path):
            qid = str(row.get("question_id") or row.get("item_id") or "")
            if not qid or qid in lookup:
                continue
            target = target_name_from_row(row)
            if target:
                lookup[qid] = target
    return lookup


def summarize_file(path: Path, target_lookup: Dict[str, str]) -> dict:
    rows = read_rows(path)
    counter: Counter = Counter()
    judge_no_answer = 0
    total_mentions = 0
    target_available = 0
    target_any_hit = 0
    target_primary_hit = 0
    for row in rows:
        mentions = list(row.get("judge_mentions_normalized") or [])
        if not mentions:
            judge_no_answer += 1
        for mention in mentions:
            counter[mention] += 1
            total_mentions += 1
        qid = str(row.get("question_id") or row.get("item_id") or "")
        target = target_lookup.get(qid)
        if target:
            target_available += 1
            if target in mentions:
                target_any_hit += 1
            if mentions and mentions[0] == target:
                target_primary_hit += 1
    variant = rows[0].get("variant") if rows else ""
    model_type = rows[0].get("model_type") if rows else ""
    metadata = rows[0].get("metadata") if rows else None
    return {
        "jsonl_path": str(path),
        "variant": variant,
        "model_type": model_type,
        "metadata": metadata,
        "questions": len(rows),
        "judge_no_answer": judge_no_answer,
        "judge_no_answer_rate": judge_no_answer / len(rows) if rows else 0.0,
        "total_geo_mentions": total_mentions,
        "target_available": target_available,
        "target_any_hit": target_any_hit,
        "target_any_hit_rate": target_any_hit / target_available if target_available else 0.0,
        "target_primary_hit": target_primary_hit,
        "target_primary_hit_rate": target_primary_hit / target_available if target_available else 0.0,
        "diversity": len(counter),
        "entropy": normalized_entropy(counter),
        "top_geo_1": counter.most_common(1)[0][0] if counter else "",
        "top_geo_1_count": counter.most_common(1)[0][1] if counter else 0,
        "top_geo_2": counter.most_common(2)[1][0] if len(counter) >= 2 else "",
        "top_geo_2_count": counter.most_common(2)[1][1] if len(counter) >= 2 else 0,
        "top_geo_3": counter.most_common(3)[2][0] if len(counter) >= 3 else "",
        "top_geo_3_count": counter.most_common(3)[2][1] if len(counter) >= 3 else 0,
    }


def main() -> int:
    args = parse_args()
    files = [Path(p) for p in args.inputs]
    target_lookup = build_target_lookup(files)
    rows = [summarize_file(path, target_lookup) for path in files]
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "jsonl_path",
                "variant",
                "model_type",
                "metadata",
                "questions",
                "judge_no_answer",
                "judge_no_answer_rate",
                "total_geo_mentions",
                "target_available",
                "target_any_hit",
                "target_any_hit_rate",
                "target_primary_hit",
                "target_primary_hit_rate",
                "diversity",
                "entropy",
                "top_geo_1",
                "top_geo_1_count",
                "top_geo_2",
                "top_geo_2_count",
                "top_geo_3",
                "top_geo_3_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[✔] Wrote {out_path}")
    print(f"[✔] Resolved targets for {len(target_lookup)} question IDs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
