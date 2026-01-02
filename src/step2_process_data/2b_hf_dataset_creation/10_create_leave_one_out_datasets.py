#!/usr/bin/env python3
"""
Create leave-one-out interleaved combined datasets from continent splits.
Each group of 3 samples contains exactly one example from each included continent
in a shuffled order, leaving the specified continent out.

Usage:
    python create_leave_one_out_datasets.py --variant with_metadata --split train --exclude africa
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

CONTINENTS = ['africa', 'asia', 'europe', 'america']

def create_leave_one_out_combined_dataset(meco_datasets_dir, metadata_variant, split, exclude_continent, seed=42):
    """Create interleaved combined dataset that omits one continent"""
    if exclude_continent not in CONTINENTS:
        raise ValueError(f"exclude_continent must be one of {CONTINENTS}")

    random.seed(seed)
    included_continents = [c for c in CONTINENTS if c != exclude_continent]

    # Load datasets for included continents
    continent_datasets = {}
    min_size = float('inf')

    for continent in included_continents:
        dataset_path = os.path.join(meco_datasets_dir, "continents", continent, metadata_variant, split)
        if os.path.exists(dataset_path):
            dataset = load_from_disk(dataset_path)
            continent_datasets[continent] = dataset
            min_size = min(min_size, len(dataset))
            print(f"{continent} {split}: {len(dataset):,} samples")
        else:
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Min dataset size (across included continents): {min_size:,}")

    batch_size = 10000
    all_indices = []

    # Build shuffled triplets for balanced interleaving
    for batch_start in tqdm(range(0, min_size, batch_size), desc="Creating index batches"):
        batch_end = min(batch_start + batch_size, min_size)

        for group_idx in range(batch_start, batch_end):
            group_indices = [(continent, group_idx) for continent in included_continents]
            random.shuffle(group_indices)
            all_indices.extend(group_indices)

    print(f"Total indices: {len(all_indices):,}")

    def generate_samples():
        for continent, idx in tqdm(all_indices, desc="Generating samples"):
            yield continent_datasets[continent][idx]

    combined_dataset = Dataset.from_generator(generate_samples)

    print(f"Combined dataset (no {exclude_continent}): {len(combined_dataset):,} samples")
    print(f"Groups of 3: {len(combined_dataset) // 3:,}")

    return combined_dataset

def main():
    parser = argparse.ArgumentParser(description="Create leave-one-out interleaved combined datasets")
    parser.add_argument("--variant", choices=["with_metadata", "without_metadata"], required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--exclude", choices=CONTINENTS, required=True, help="Continent to leave out")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config = Config()
    meco_datasets_dir = os.path.join(config.TRAINING_DATA_BASE, "meco_datasets")

    print(f"Creating leave-one-out combined dataset (no {args.exclude}): {args.variant}/{args.split}")
    print("=" * 60)

    combined_dataset = create_leave_one_out_combined_dataset(
        meco_datasets_dir, args.variant, args.split, args.exclude, args.seed
    )

    # Save combined dataset
    output_path = os.path.join(meco_datasets_dir, f"combined_no_{args.exclude}", args.variant, args.split)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    combined_dataset.save_to_disk(output_path)
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
