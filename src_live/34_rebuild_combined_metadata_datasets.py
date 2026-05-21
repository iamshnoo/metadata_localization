#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/path/to/workspace/.cache/huggingface")
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(os.environ["HF_HOME"], "datasets"))
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

from datasets import load_from_disk


_CONTINENT_DATASET = None
_CONTINENT_PATH = None


def split_prefix_and_body(text):
    marker = "TITLE: "
    idx = text.find(marker)
    if idx < 0:
        raise ValueError("TITLE marker not found")
    prefix = text[:idx].strip()
    body = text[idx:]
    return prefix, body


def load_continent_dataset(path):
    global _CONTINENT_DATASET, _CONTINENT_PATH
    if _CONTINENT_DATASET is None or _CONTINENT_PATH != path:
        _CONTINENT_DATASET = load_from_disk(path)
        _CONTINENT_PATH = path
    return _CONTINENT_DATASET


def build_with_metadata_batch(batch, indices, continent_path, validate_body):
    continent_ds = load_continent_dataset(continent_path)
    continent_texts = continent_ds[indices]["text"]
    outputs = []
    for country_text, continent_text in zip(batch["text"], continent_texts):
        country_prefix, country_body = split_prefix_and_body(country_text)
        continent_prefix, continent_body = split_prefix_and_body(continent_text)
        if validate_body and country_body != continent_body:
            raise ValueError("Country and continent source rows are not aligned")
        outputs.append(f"{country_prefix}\n{continent_prefix}\n\n{country_body}")
    return {"text": outputs}


def build_without_metadata_batch(batch):
    return {"text": [split_prefix_and_body(text)[1] for text in batch["text"]]}


def validate_sources(country_ds, continent_ds, split):
    if len(country_ds) != len(continent_ds):
        raise ValueError(f"{split}: source sizes differ: {len(country_ds)} vs {len(continent_ds)}")
    if len(country_ds) == 0:
        raise ValueError(f"{split}: source dataset is empty")
    sample_indices = sorted(set([0, len(country_ds) // 2, len(country_ds) - 1]))
    for idx in sample_indices:
        _, country_body = split_prefix_and_body(country_ds[idx]["text"])
        _, continent_body = split_prefix_and_body(continent_ds[idx]["text"])
        if country_body != continent_body:
            raise ValueError(f"{split}: source rows differ at index {idx}")


def rebuild_split(args, split):
    country_path = args.country_root / split
    continent_path = args.continent_root / split
    with_output = args.with_output_root / split
    without_output = args.without_output_root / split

    country_ds = load_from_disk(str(country_path))
    continent_ds = load_from_disk(str(continent_path))
    validate_sources(country_ds, continent_ds, split)

    print(f"[{split}] rows={len(country_ds):,}")
    print(f"[{split}] writing {with_output}")
    with_ds = country_ds.map(
        build_with_metadata_batch,
        batched=True,
        with_indices=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        fn_kwargs={
            "continent_path": str(continent_path),
            "validate_body": args.validate_all_rows,
        },
        desc=f"{split}: COUNTRY+CONTINENT",
    )
    with_output.parent.mkdir(parents=True, exist_ok=True)
    with_ds.save_to_disk(str(with_output), num_proc=args.num_proc)

    print(f"[{split}] writing {without_output}")
    without_ds = country_ds.map(
        build_without_metadata_batch,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        desc=f"{split}: no metadata",
    )
    without_output.parent.mkdir(parents=True, exist_ok=True)
    without_ds.save_to_disk(str(without_output), num_proc=args.num_proc)

    print(f"[{split}] done")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild aligned combined with/without metadata HF datasets from the "
            "surviving country-only and continent-only combined datasets."
        )
    )
    parser.add_argument(
        "--country-root",
        type=Path,
        default=Path("/path/to/metacul/training_data/meco_datasets/combined_only_country/with_metadata"),
    )
    parser.add_argument(
        "--continent-root",
        type=Path,
        default=Path("/path/to/metacul/training_data/meco_datasets/combined_only_continent/with_metadata"),
    )
    parser.add_argument(
        "--with-output-root",
        type=Path,
        default=Path("/path/to/workspace/pretrain/datasets/raw-combined-with-metadata"),
    )
    parser.add_argument(
        "--without-output-root",
        type=Path,
        default=Path("/path/to/workspace/pretrain/datasets/raw-combined-without-metadata"),
    )
    parser.add_argument("--split", choices=["train", "validation", "test"], action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument(
        "--validate-all-rows",
        action="store_true",
        help="Check body equality for every row while rebuilding with_metadata.",
    )
    args = parser.parse_args()

    splits = args.split or ["train", "validation", "test"]
    for split in splits:
        rebuild_split(args, split)


if __name__ == "__main__":
    main()
