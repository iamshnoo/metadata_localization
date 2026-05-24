#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

from config import (
    COUNTRY_TO_CONTINENT,
    CONTRAST_COUNTRY_MAP,
    DEFAULT_BATCH_ENDPOINT,
    DEFAULT_CORE_TARGET,
    DEFAULT_GENERATION_BATCH_SIZE,
    DEFAULT_MODEL,
    DEFAULT_OVERSAMPLE,
    DEFAULT_YEAR_RANGE,
    GENERATION_ANGLES,
    GENERATION_SPECS,
    TOPIC_FOCUS_AREAS,
    TOPIC_LABELS,
    YEAR_BUCKETS,
    per_topic_quota,
)


def iter_requests(
    per_country_target: int,
    oversample: float,
    batch_size: int,
    model: str,
) -> Iterator[Tuple[Dict, Dict]]:
    target = int(round(per_country_target * oversample))
    topic_quota = per_topic_quota(target)
    for split_key, spec in GENERATION_SPECS.items():
        developer_prompt = spec.developer_prompt_path.read_text(encoding="utf-8").strip()
        for country, continent in COUNTRY_TO_CONTINENT.items():
            contrast_country = CONTRAST_COUNTRY_MAP[country]
            for topic, quota in topic_quota.items():
                remaining = quota
                batch_idx = 0
                while remaining > 0:
                    n_items = min(batch_size, remaining)
                    shard_year_range = YEAR_BUCKETS[batch_idx % len(YEAR_BUCKETS)]
                    shard_focus = TOPIC_FOCUS_AREAS[topic][batch_idx % len(TOPIC_FOCUS_AREAS[topic])]
                    shard_angle = GENERATION_ANGLES[(batch_idx // len(YEAR_BUCKETS)) % len(GENERATION_ANGLES)]
                    shard_count = (quota + batch_size - 1) // batch_size
                    custom_id = f"{split_key}__{country.lower().replace(' ', '_')}__{topic}__b{batch_idx:03d}"
                    user_prompt = (
                        f"Build {n_items} items for LocalNewsQA-Core.\\n"
                        f"Split type: {split_key}.\\n"
                        f"Target country: {country}.\\n"
                        f"Target continent: {continent}.\\n"
                        f"Contrast country: {contrast_country}.\\n"
                        f"Topic: {TOPIC_LABELS[topic]}.\\n"
                        f"Global year range: {DEFAULT_YEAR_RANGE}.\\n"
                        f"Shard index: {batch_idx + 1} of {shard_count}.\\n"
                        f"Prioritize coverage from this year window: {shard_year_range}.\\n"
                        f"Prioritize this factual focus: {shard_focus}.\\n"
                        f"Question-writing angle: {shard_angle}.\\n"
                        "This shard must avoid overlap with other shards for the same country/topic/split. "
                        "Do not reuse the same person, office, institution, stadium, broadcaster, holiday, "
                        "or event as the anchor of multiple questions unless absolutely necessary. "
                        "Use unique questions only. Keep the JSON compact and valid."
                    )
                    request = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": DEFAULT_BATCH_ENDPOINT,
                        "body": {
                            "model": model,
                            "temperature": 0.4,
                            "max_completion_tokens": 4096,
                            "messages": [
                                {"role": "developer", "content": developer_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            "response_format": {"type": "json_object"},
                        },
                    }
                    meta = {
                        "custom_id": custom_id,
                        "split_key": split_key,
                        "split_name": spec.split_name,
                        "country": country,
                        "continent": continent,
                        "contrast_country": contrast_country,
                        "topic": topic,
                        "batch_index": batch_idx,
                        "shard_year_range": shard_year_range,
                        "shard_focus": shard_focus,
                        "shard_angle": shard_angle,
                        "requested_items": n_items,
                        "model": model,
                    }
                    yield request, meta
                    remaining -= n_items
                    batch_idx += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Build OpenAI Batch API requests for LocalNewsQA-Core generation.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--per-country-target", type=int, default=DEFAULT_CORE_TARGET)
    parser.add_argument("--oversample", type=float, default=DEFAULT_OVERSAMPLE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_GENERATION_BATCH_SIZE)
    parser.add_argument("--out-requests", default=str(Path(__file__).resolve().parent / "generation_requests.jsonl"))
    parser.add_argument("--out-manifest", default=str(Path(__file__).resolve().parent / "generation_manifest.csv"))
    args = parser.parse_args()

    requests_path = Path(args.out_requests)
    manifest_path = Path(args.out_manifest)
    requests_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(iter_requests(args.per_country_target, args.oversample, args.batch_size, args.model))
    with requests_path.open("w", encoding="utf-8") as f:
        for request, _ in rows:
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0][1].keys()))
        writer.writeheader()
        for _, meta in rows:
            writer.writerow(meta)

    print(f"Wrote {len(rows)} generation requests to {requests_path}")
    print(f"Wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
