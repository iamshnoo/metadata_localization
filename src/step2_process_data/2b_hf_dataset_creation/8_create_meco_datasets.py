#!/usr/bin/env python3
"""
Create MECO-style datasets with and without metadata for pretraining.
MECO paper - https://arxiv.org/pdf/2501.01956

Usage:
    python create_meco_datasets.py --split-type continents --split-name africa
    python create_meco_datasets.py --split-type novel-concept --split-name pivot_2018
    python create_meco_datasets.py --split-type concept-change --split-name politics
    python create_meco_datasets.py --split-type all

    # Metadata ablations for continent splits (examples)
    python create_meco_datasets.py --split-type continents --metadata-ablation url_only
    python create_meco_datasets.py --split-type continents --split-name africa --metadata-ablation url_country
    python create_meco_datasets.py --split-type continents --metadata-ablation url_continent
"""

import sys
import argparse
import os
from pathlib import Path
from datasets import Dataset, DatasetDict, Features, Value
import gc

sys.path.append(str(Path(__file__).parent))
from config import Config

METADATA_ABLATION_CHOICES = ["full", "url_only", "url_country", "url_continent"]

class MECODatasetCreator:
    def __init__(self):
        self.config = Config()
        self.hf_datasets_dir = os.path.join(self.config.TRAINING_DATA_BASE, "hf_datasets")
        self.meco_datasets_dir = os.path.join(self.config.TRAINING_DATA_BASE, "meco_datasets")
        os.makedirs(self.meco_datasets_dir, exist_ok=True)

        # Single text field for pretraining
        self.features = Features({'text': Value('string')})

    def format_with_metadata(self, record, split_type, metadata_ablation="full"):
        """Format record with metadata following MECO template (with optional ablation for continents)"""
        metadata_parts = []

        if split_type == 'continents':
            # Apply ablations only for continent splits
            if metadata_ablation == "url_only":
                metadata_parts.append(f"URL: {record['url']}")
            elif metadata_ablation == "url_country":
                metadata_parts.extend([
                    f"URL: {record['url']}",
                    f"COUNTRY: {record['country']}",
                ])
            elif metadata_ablation == "url_continent":
                metadata_parts.extend([
                    f"URL: {record['url']}",
                    f"CONTINENT: {record['continent']}",
                ])
            else:  # full metadata
                metadata_parts.extend([
                    f"URL: {record['url']}",
                    f"COUNTRY: {record['country']}",
                    f"CONTINENT: {record['continent']}"
                ])

        elif split_type == 'concept-change':
            metadata_parts.extend([
                f"URL: {record['url']}",
                f"COUNTRY: {record['country']}",
                f"CONTINENT: {record['continent']}"
            ])

        elif split_type == 'novel-concept':
            metadata_parts.extend([
                f"URL: {record['url']}",
                f"COUNTRY: {record['country']}",
                f"CONTINENT: {record['continent']}",
                f"YEAR: {record['year']}"
            ])

        metadata_str = "\n".join(metadata_parts)
        content = f"TITLE: {record['title']}\n\nCONTENT: {record['content']}"

        return f"{metadata_str}\n\n{content}"

    def format_without_metadata(self, record):
        """Format record without metadata"""
        return f"TITLE: {record['title']}\n\nCONTENT: {record['content']}"

    def create_meco_generator(self, dataset, split_type, with_metadata=True, metadata_ablation="full"):
        """Generator that yields MECO-formatted records"""
        for record in dataset:
            if with_metadata:
                text = self.format_with_metadata(record, split_type, metadata_ablation)
            else:
                text = self.format_without_metadata(record)

            yield {'text': text}

    def process_split(self, input_path, output_base, split_type, split_name, metadata_ablation="full"):
        """Process a single split to create with/without metadata versions"""
        print(f"Processing {split_name} ({split_type})")

        # Load existing HF dataset
        dataset_dict = DatasetDict.load_from_disk(input_path)

        for split in ['train', 'validation', 'test']:  # Updated split names
            if split not in dataset_dict:
                continue

            print(f"  Creating {split} split...")
            original_dataset = dataset_dict[split]

            # Create with metadata version
            with_meta_dataset = Dataset.from_generator(
                lambda: self.create_meco_generator(original_dataset, split_type, True, metadata_ablation),
                features=self.features
            )

            # Create without metadata version
            without_meta_dataset = Dataset.from_generator(
                lambda: self.create_meco_generator(original_dataset, split_type, False),
                features=self.features
            )

            # Save datasets
            with_meta_path = os.path.join(output_base, "with_metadata", split)
            without_meta_path = os.path.join(output_base, "without_metadata", split)

            os.makedirs(with_meta_path, exist_ok=True)
            os.makedirs(without_meta_path, exist_ok=True)

            with_meta_dataset.save_to_disk(with_meta_path)
            without_meta_dataset.save_to_disk(without_meta_path)

            print(f"    With metadata: {len(with_meta_dataset):,} samples")
            print(f"    Without metadata: {len(without_meta_dataset):,} samples")

            # Cleanup
            del with_meta_dataset, without_meta_dataset
            gc.collect()

    def _continent_output_dir(self, continent, metadata_ablation):
        """Construct output directory name for continent splits based on ablation."""
        base = continent.lower()
        if metadata_ablation == "url_only":
            base = f"url-only-{base}"
        elif metadata_ablation == "url_country":
            base = f"url-country-{base}"
        elif metadata_ablation == "url_continent":
            base = f"url-continent-{base}"
        return os.path.join(self.meco_datasets_dir, "continents", base)

    def process_continents(self, specific_continent=None, metadata_ablation="full"):
        """Process continent splits"""
        continents = ['Africa', 'Asia', 'Europe', 'America']  # Use capitalized names
        if specific_continent:
            # Handle both lowercase input and capitalized directory names
            continent_map = {'africa': 'Africa', 'asia': 'Asia', 'europe': 'Europe', 'america': 'America'}
            target = continent_map.get(specific_continent.lower(), specific_continent)
            continents = [c for c in continents if c == target]

        for continent in continents:
            # Try lowercase first, then capitalized for HF dataset path
            input_path_lower = os.path.join(self.hf_datasets_dir, "continents", continent.lower())
            input_path_cap = os.path.join(self.hf_datasets_dir, "continents", continent)

            if os.path.exists(input_path_lower):
                input_path = input_path_lower
            elif os.path.exists(input_path_cap):
                input_path = input_path_cap
            else:
                print(f"Warning: No HF dataset found for {continent} at either:")
                print(f"  {input_path_lower}")
                print(f"  {input_path_cap}")
                continue

            output_path = self._continent_output_dir(continent, metadata_ablation)
            self.process_split(input_path, output_path, "continents", continent.lower(), metadata_ablation)

    def process_novel_concept(self, specific_pivot=None):
        """Process novel concept splits"""
        pivots = [f"pivot_{year}" for year in self.config.PIVOT_YEARS]
        if specific_pivot:
            pivots = [p for p in pivots if p == specific_pivot]

        for pivot in pivots:
            input_path = os.path.join(self.hf_datasets_dir, "novel_concept", pivot)
            output_path = os.path.join(self.meco_datasets_dir, "novel_concept", pivot)

            if os.path.exists(input_path):
                self.process_split(input_path, output_path, "novel-concept", pivot)

    def process_concept_change(self, specific_theme=None):
        """Process concept change splits"""
        concept_dir = os.path.join(self.hf_datasets_dir, "concept_change")
        if not os.path.exists(concept_dir):
            return

        themes = [d for d in os.listdir(concept_dir) if os.path.isdir(os.path.join(concept_dir, d))]
        if specific_theme:
            theme_key = specific_theme.lower().replace(' ', '_')
            themes = [t for t in themes if t.lower() == theme_key]

        for theme in themes:
            input_path = os.path.join(concept_dir, theme)
            output_path = os.path.join(self.meco_datasets_dir, "concept_change", theme)

            self.process_split(input_path, output_path, "concept-change", theme)

def main():
    parser = argparse.ArgumentParser(description="Create MECO-style datasets")
    parser.add_argument("--split-type", choices=["all", "continents", "novel-concept", "concept-change"],
                       default="all")
    parser.add_argument("--split-name", help="Specific split name to process")
    parser.add_argument("--metadata-ablation", choices=METADATA_ABLATION_CHOICES, default="full",
                       help="Metadata fields to include for continent splits")

    args = parser.parse_args()

    creator = MECODatasetCreator()

    try:
        if args.split_type in ["all", "continents"]:
            creator.process_continents(args.split_name, args.metadata_ablation)

        if args.split_type in ["all", "novel-concept"]:
            creator.process_novel_concept(args.split_name)

        if args.split_type in ["all", "concept-change"]:
            creator.process_concept_change(args.split_name)

        print(f"\n✅ MECO datasets created in: {creator.meco_datasets_dir}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
