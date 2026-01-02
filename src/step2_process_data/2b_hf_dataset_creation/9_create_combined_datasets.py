#!/usr/bin/env python3
"""
Create interleaved combined datasets from continent splits.
Every 4 items contains exactly 1 sample from each continent in shuffled order.

Usage:
    python 9_create_combined_datasets.py --variant with_metadata --split train
    python 9_create_combined_datasets.py --variant without_metadata --split validation

    # Metadata ablations (continents produced by step 8 with ablation flag)
    python 9_create_combined_datasets.py --variant with_metadata --split train --ablation only_url
    python 9_create_combined_datasets.py --variant with_metadata --split validation --ablation only_url_country
    python 9_create_combined_datasets.py --variant with_metadata --split test --ablation only_url_continent
"""

import os
import random
import argparse
from datasets import Dataset, load_from_disk
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent))
from config import Config

ABLATION_CHOICES = ["none", "only_url", "only_url_country", "only_url_continent"]


def _continent_dir(continent, ablation):
    """Return continent directory name based on ablation flag."""
    if ablation == "only_url":
        return f"url-only-{continent}"
    if ablation == "only_url_country":
        return f"url-country-{continent}"
    if ablation == "only_url_continent":
        return f"url-continent-{continent}"
    return continent


def _combined_dir(ablation):
    """Return combined output directory name based on ablation flag."""
    if ablation == "only_url":
        return "combined_only_url"
    if ablation == "only_url_country":
        return "combined_only_url_country"
    if ablation == "only_url_continent":
        return "combined_only_url_continent"
    return "combined"


def create_interleaved_combined_dataset(meco_datasets_dir, metadata_variant, split, ablation="none", seed=42):
    """Create interleaved combined dataset with balanced continent representation"""
    random.seed(seed)
    
    continents = ['africa', 'asia', 'europe', 'america']
    
    # Load all continent datasets for this variant and split
    continent_datasets = {}
    min_size = float('inf')
    
    for continent in continents:
        continent_dir = _continent_dir(continent, ablation)
        dataset_path = os.path.join(meco_datasets_dir, "continents", continent_dir, metadata_variant, split)
        if os.path.exists(dataset_path):
            dataset = load_from_disk(dataset_path)
            continent_datasets[continent] = dataset
            min_size = min(min_size, len(dataset))
            print(f"{continent_dir} {split}: {len(dataset):,} samples")
        else:
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    print(f"Min dataset size: {min_size:,}")

    # Process in batches to avoid memory issues
    batch_size = 10000
    all_indices = []

    # Create shuffled indices for each group of 4
    for batch_start in tqdm(range(0, min_size, batch_size), desc="Creating index batches"):
        batch_end = min(batch_start + batch_size, min_size)

        for group_idx in range(batch_start, batch_end):
            # Create indices for this group and shuffle
            group_indices = [(continent, group_idx) for continent in continents]
            random.shuffle(group_indices)
            all_indices.extend(group_indices)

    print(f"Total indices: {len(all_indices):,}")

    # Create dataset from indices
    def generate_samples():
        for continent, idx in tqdm(all_indices, desc="Generating samples"):
            yield continent_datasets[continent][idx]

    combined_dataset = Dataset.from_generator(generate_samples)

    print(f"Combined dataset: {len(combined_dataset):,} samples")
    print(f"Groups of 4: {len(combined_dataset) // 4:,}")

    return combined_dataset

def main():
    parser = argparse.ArgumentParser(description="Create interleaved combined datasets")
    parser.add_argument("--variant", choices=["with_metadata", "without_metadata"], required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--ablation", choices=ABLATION_CHOICES, default="none",
                        help="Metadata ablation variant for continent sources (produced in step 8)")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    config = Config()
    meco_datasets_dir = os.path.join(config.TRAINING_DATA_BASE, "meco_datasets")
    
    print(f"Creating combined dataset: {args.variant}/{args.split} (ablation: {args.ablation})")
    print("=" * 50)
    
    combined_dataset = create_interleaved_combined_dataset(
        meco_datasets_dir, args.variant, args.split, args.ablation, args.seed
    )
    
    # Save combined dataset
    combined_dir = _combined_dir(args.ablation)
    output_path = os.path.join(meco_datasets_dir, combined_dir, args.variant, args.split)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    combined_dataset.save_to_disk(output_path)
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
