#!/usr/bin/env python3
"""
Create combined metadata ablation datasets directly from combined/with_metadata.

Supported variants:
- only_country
- only_continent
"""

import argparse
import os
from pathlib import Path

HF_HOME_DEFAULT = "/scratch/amukher6/.cache/huggingface"
os.environ.setdefault("HF_HOME", HF_HOME_DEFAULT)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(os.environ["HF_HOME"], "datasets"))
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

from datasets import load_from_disk


def split_metadata_and_body(text: str):
    lines = text.splitlines()
    country = None
    continent = None
    body_start = None

    for idx, line in enumerate(lines):
        if line.startswith("COUNTRY: "):
            country = line[len("COUNTRY: ") :]
        elif line.startswith("CONTINENT: "):
            continent = line[len("CONTINENT: ") :]
        elif line.startswith("TITLE: "):
            body_start = idx
            break

    if body_start is None:
        raise ValueError("Could not find TITLE section in sample.")
    if country is None:
        raise ValueError("Could not find COUNTRY field in sample.")
    if continent is None:
        raise ValueError("Could not find CONTINENT field in sample.")

    body = "\n".join(lines[body_start:])
    return country, continent, body


def convert_text(text: str, ablation: str) -> str:
    country, continent, body = split_metadata_and_body(text)
    if ablation == "only_country":
        return f"COUNTRY: {country}\n\n{body}"
    if ablation == "only_continent":
        return f"CONTINENT: {continent}\n\n{body}"
    raise ValueError(f"Unsupported ablation: {ablation}")


def main():
    parser = argparse.ArgumentParser(description="Create combined metadata ablation datasets from combined/with_metadata.")
    parser.add_argument("--ablation", choices=["only_country", "only_continent"], required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--num-proc", type=int, default=8)
    args = parser.parse_args()

    base = Path("/scratch/amukher6/metacul/training_data/meco_datasets")
    src_path = base / "combined" / "with_metadata" / args.split

    out_dir_name = "combined_only_country" if args.ablation == "only_country" else "combined_only_continent"
    out_path = base / out_dir_name / "with_metadata" / args.split
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_from_disk(str(src_path))
    out = ds.map(
        lambda ex: {"text": convert_text(ex["text"], args.ablation)},
        num_proc=args.num_proc,
        desc=f"Converting {args.split} -> {args.ablation}",
    )
    out.save_to_disk(str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
