#!/usr/bin/env python3

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

from datasets import Dataset

from config import DEFAULT_CORE_FALLBACK, DEFAULT_CORE_TARGET, SPLIT_FAMILY, TOPICS


def normalize_question(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return text


def parse_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


def load_verifications(batch_output: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with batch_output.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            custom_id = row.get("custom_id")
            body = ((row.get("response") or {}).get("body") or {})
            choices = body.get("choices") or []
            if not choices:
                continue
            content = (((choices[0] or {}).get("message") or {}).get("content")) or ""
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            try:
                out[custom_id] = parse_json_object(content)
            except Exception:
                out[custom_id] = {"final_decision": "reject", "notes": "parse_error"}
    return out


def attach_verification(candidates_path: Path, verification_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with candidates_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            custom_id = f"verify__{item['split_type']}__{idx:08d}"
            item["verification"] = verification_map.get(custom_id, {"final_decision": "reject", "notes": "missing_verification"})
            item["verification_status"] = item["verification"].get("final_decision", "reject")
            items.append(item)
    return items


def choose_target(items_by_country_split: Dict[Tuple[str, str], List[Dict[str, Any]]], target: int, fallback: int) -> int:
    min_count = min((len(v) for v in items_by_country_split.values()), default=0)
    if min_count >= target:
        return target
    if min_count >= fallback:
        return fallback
    raise SystemExit(f"Not enough verified items for fallback target; minimum available is {min_count}")


def dedupe_and_select(items: List[Dict[str, Any]], target_per_country_split: int) -> List[Dict[str, Any]]:
    by_bucket: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    seen = set()  # type: Set[Tuple[str, str, str]]
    for item in items:
        if item.get("verification_status") != "pass":
            continue
        key = (item["country"], item["split_type"], normalize_question(item["question"]))
        if key in seen:
            continue
        seen.add(key)
        by_bucket[(item["country"], item["split_type"])].append(item)

    selected: List[Dict[str, Any]] = []
    for bucket_key, bucket in sorted(by_bucket.items()):
        bucket.sort(key=lambda x: (TOPICS.index(x.get("topic", TOPICS[0])) if x.get("topic") in TOPICS else 999, x.get("year", 9999), x.get("question", "")))
        counts_by_topic = defaultdict(int)
        for item in bucket:
            if len([r for r in selected if r["country"] == bucket_key[0] and r["split_type"] == bucket_key[1]]) >= target_per_country_split:
                break
            topic = item.get("topic", TOPICS[0])
            topic_cap = max(1, target_per_country_split // len(TOPICS) + 2)
            if counts_by_topic[topic] >= topic_cap:
                continue
            counts_by_topic[topic] += 1
            selected.append(item)
    return selected


def write_report(items: Iterable[Dict[str, Any]], path: Path) -> None:
    counts = defaultdict(int)
    for item in items:
        counts[(item["country"], item["split_type"])] += 1
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["country", "split_type", "count"])
        for (country, split_type), count in sorted(counts.items()):
            writer.writerow([country, split_type, count])


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize LocalNewsQA-Core from verified candidates.")
    parser.add_argument("--candidates-jsonl", required=True)
    parser.add_argument("--verification-batch-output", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--jsonl-out", required=True)
    parser.add_argument("--report-csv", required=True)
    parser.add_argument("--target-per-country", type=int, default=DEFAULT_CORE_TARGET)
    parser.add_argument("--fallback-per-country", type=int, default=DEFAULT_CORE_FALLBACK)
    args = parser.parse_args()

    verification_map = load_verifications(Path(args.verification_batch_output))
    items = attach_verification(Path(args.candidates_jsonl), verification_map)

    buckets = defaultdict(list)
    for item in items:
        if item.get("verification_status") == "pass":
            buckets[(item["country"], item["split_type"])].append(item)

    selected_target = choose_target(buckets, args.target_per_country, args.fallback_per_country)
    selected = dedupe_and_select(items, selected_target)

    for idx, item in enumerate(selected):
        item["id"] = f"localnewsqa_core_{idx:07d}"
        item["split_family"] = SPLIT_FAMILY
        item["evidence_urls"] = [ev.get("url", "") for ev in item.get("evidence_pack", []) if ev.get("url")]
        item["evidence_source_type"] = sorted({ev.get("source_type", "") for ev in item.get("evidence_pack", []) if ev.get("source_type")})
        item["verifier_model"] = "gpt-5.4"

    jsonl_out = Path(args.jsonl_out)
    jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_out.open("w", encoding="utf-8") as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    dataset = Dataset.from_list(selected)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))
    write_report(selected, Path(args.report_csv))

    print(f"Selected target per country/split: {selected_target}")
    print(f"Saved {len(selected)} rows to {output_dir}")
    print(f"Wrote JSONL to {jsonl_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
