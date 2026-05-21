#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_items(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data).__name__}")
    return data


def normalize_record(
    item: Dict[str, Any], continent: str, generated_by: str, source_path: Path
) -> Dict[str, Any]:
    required = ["question", "options", "correct_answer", "distractors", "country"]
    missing = [k for k in required if k not in item]
    if missing:
        raise KeyError(f"Missing keys {missing} in {source_path}")

    return {
        "question": item["question"],
        "options": item["options"],
        "correct_answer": item["correct_answer"],
        "distractors": item["distractors"],
        "country": item["country"],
        "continent": continent,
        "generated_by": generated_by,
    }


def build_records(input_root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for continent_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        continent = continent_dir.name
        for json_path in sorted(continent_dir.glob("*.json")):
            generated_by = json_path.stem
            items = load_items(json_path)
            for item in items:
                records.append(
                    normalize_record(item, continent, generated_by, json_path)
                )
    return records


def write_jsonl(records: Iterable[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def print_report(records: Iterable[Dict[str, Any]]) -> None:
    by_continent: Counter[str] = Counter()
    by_generator: Counter[str] = Counter()
    total = 0
    for rec in records:
        total += 1
        by_continent[rec["continent"]] += 1
        by_generator[rec["generated_by"]] += 1

    print("Report")
    print(f"Total records: {total}")
    print("By continent:")
    for name, count in by_continent.most_common():
        print(f"  {name}: {count}")
    print("By generated_by:")
    for name, count in by_generator.most_common():
        print(f"  {name}: {count}")


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build a HuggingFace dataset from metacul/qa_data JSON files."
    )
    parser.add_argument(
        "--input-root",
        default=str(script_dir),
        help="Root folder containing continent subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(script_dir / "hf_dataset"),
        help="Output directory for Dataset.save_to_disk().",
    )
    parser.add_argument(
        "--jsonl-out",
        default=str(script_dir / "hf_dataset.jsonl"),
        help="Path to write JSONL export of the dataset.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print a summary report by continent and generator.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload dataset to HuggingFace Hub (YOUR_HF_USERNAME/qa_metacul).",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    jsonl_out = Path(args.jsonl_out)

    records = build_records(input_root)
    if not records:
        raise SystemExit(f"No records found under {input_root}")

    try:
        from datasets import Dataset
    except ImportError as exc:
        raise SystemExit(
            "datasets library not found. Install with: pip install datasets"
        ) from exc

    dataset = Dataset.from_list(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))
    write_jsonl(records, jsonl_out)

    if args.report:
        print_report(records)

    if args.upload:
        repo_id = "YOUR_HF_USERNAME/qa_metacul"
        dataset.push_to_hub(repo_id)
        print(f"Uploaded dataset to {repo_id}")

    print(f"Saved {len(dataset)} rows to {output_dir}")
    print(f"Wrote JSONL to {jsonl_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
