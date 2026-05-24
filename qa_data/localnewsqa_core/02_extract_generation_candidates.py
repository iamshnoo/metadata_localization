#!/usr/bin/env python3

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def load_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["custom_id"]: row for row in csv.DictReader(f)}


def parse_generation_payload(text: str) -> List[Dict[str, Any]]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        obj_start = text.find("{")
        obj_end = text.rfind("}")
        arr_start = text.find("[")
        arr_end = text.rfind("]")
        if obj_start != -1 and obj_end != -1:
            data = json.loads(text[obj_start : obj_end + 1])
        elif arr_start != -1 and arr_end != -1:
            data = json.loads(text[arr_start : arr_end + 1])
        else:
            raise
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        items = data.get("items")
        if isinstance(items, list):
            return items
    raise ValueError("Expected a JSON object with an 'items' list or a JSON array")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract LocalNewsQA-Core generation candidates from batch output JSONL.")
    parser.add_argument("--batch-output", required=True)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--out-jsonl", required=True)
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest_csv))
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with open(args.batch_output, "r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as out:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            custom_id = str(row.get("custom_id", "")).strip()
            meta = manifest.get(custom_id)
            if meta is None:
                continue
            body = ((row.get("response") or {}).get("body") or {})
            choices = body.get("choices") or []
            if not choices:
                continue
            content = (((choices[0] or {}).get("message") or {}).get("content")) or ""
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            items = parse_generation_payload(content)
            for idx, item in enumerate(items):
                item["split_name"] = meta["split_name"]
                item["split_type"] = meta["split_key"]
                item["split_family"] = "LocalNewsQA-Core"
                item["country"] = item.get("country") or meta["country"]
                item["continent"] = item.get("continent") or meta["continent"]
                item["contrast_country"] = item.get("contrast_country") or meta["contrast_country"]
                item["generation_shard_year_range"] = meta.get("shard_year_range", "")
                item["generation_shard_focus"] = meta.get("shard_focus", "")
                item["generation_shard_angle"] = meta.get("shard_angle", "")
                item["generator_model"] = meta["model"]
                item["generation_custom_id"] = custom_id
                item["generation_item_index"] = idx
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept += 1
    print(f"Wrote {kept} candidate items to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
