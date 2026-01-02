#!/usr/bin/env python3
"""
Ultra-fast data splitter that creates splits with doc IDs and metadata only.

Creates individual train/valid/test splits + global test sets for fair comparison.

For each split type:
- Individual splits: train/valid/test (test size = 10000/num_splits)
- Global test set: combines all individual test sets (total = 10000)

Usage:
    python data_splitter_ids_only.py --split-type continents --seed 42
    python data_splitter_ids_only.py --split-type concept-change --seed 42
    python data_splitter_ids_only.py --split-type novel-concept --seed 42
    python data_splitter_ids_only.py --split-type all --seed 42

    python data_splitter_ids_only.py --show-splits
"""

import sys
import argparse
import logging
import os
import pickle
import json
import random
import gc
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import Config

class IDsOnlyDataSplitter:
    def __init__(self, seed=42, train_valid_ratio=0.9, global_test_size=10000):
        self.config = Config()
        self.seed = seed
        self.train_valid_ratio = train_valid_ratio
        self.global_test_size = global_test_size
        random.seed(seed)

        # Setup logging
        self.setup_logging()

        # Output directories
        self.training_data_base = self.config.TRAINING_DATA_BASE
        self.individual_splits_dir = os.path.join(self.training_data_base, "individual_splits")
        self.global_test_dir = os.path.join(self.training_data_base, "global_test_sets")

        # Cache directory
        self.cache_dir = os.path.join(self.training_data_base, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.individual_splits_dir, exist_ok=True)
        os.makedirs(self.global_test_dir, exist_ok=True)

        # Meta-index file
        self.meta_index_file = os.path.join(self.cache_dir, f"meta_index_{seed}.pkl")

        self.logger.info(f"Initialized IDs-only data splitter with seed {seed}")
        self.logger.info(f"Global test size: {global_test_size}, Train/Valid ratio: {train_valid_ratio}")

    def setup_logging(self):
        """Setup logging"""
        log_dir = os.path.join(self.config.BASE, "metacul/logs")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "data_splitter_ids_only.log")

        # Create logger
        self.logger = logging.getLogger("data_splitter_ids_only")
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def load_meta_index(self):
        """Load existing meta-index"""
        if not os.path.exists(self.meta_index_file):
            raise FileNotFoundError(f"Meta-index not found: {self.meta_index_file}")

        try:
            with open(self.meta_index_file, 'rb') as f:
                meta_index = pickle.load(f)

            file_size = os.path.getsize(self.meta_index_file) / (1024 * 1024)  # MB
            self.logger.info(f"📂 Loaded meta-index ({file_size:.1f} MB)")

            stats = meta_index['stats']
            self.logger.info(f"  📄 Total documents: {stats['total_documents']:,}")
            self.logger.info(f"  🌍 Continents: {list(stats['continent_counts'].keys())}")

            return meta_index

        except Exception as e:
            self.logger.error(f"Failed to load meta-index: {e}")
            raise

    def create_doc_metadata_record(self, doc_id, meta_index):
        """Create lightweight metadata record for a document"""
        doc_info = meta_index['documents'][doc_id]

        return {
            'doc_id': doc_id,
            'year': doc_info['year'],
            'country': doc_info['country'],
            'continent': doc_info['continent'],
            'file_path': doc_info['file_path'],
            'file_index': doc_info['file_index'],
            'has_themes': doc_info['has_themes'],
            'dominant_theme': doc_info['dominant_theme']
        }

    def create_continent_splits_ids_only(self, meta_index):
        """Create continent splits with individual train/valid/test + global test set"""
        self.logger.info("🌍 Creating continent-based splits ...")

        continents = list(meta_index['stats']['continent_counts'].keys())
        num_continents = len(continents)
        test_size_per_continent = self.global_test_size // num_continents

        self.logger.info(f"Continents: {continents}")
        self.logger.info(f"Test size per continent: {test_size_per_continent}")

        global_test_records = []

        for continent in continents:
            self.logger.info(f"Processing {continent}...")

            # Create directories
            continent_dir = os.path.join(self.individual_splits_dir, "continents", continent)
            train_dir = os.path.join(continent_dir, "train")
            valid_dir = os.path.join(continent_dir, "valid")
            test_dir = os.path.join(continent_dir, "test")
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(valid_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            # Get all document IDs from this continent
            all_doc_ids = meta_index['by_continent'][continent]
            random.shuffle(all_doc_ids)

            # Split: test first, then train/valid from remainder
            test_doc_ids = all_doc_ids[:test_size_per_continent]
            remaining_doc_ids = all_doc_ids[test_size_per_continent:]

            # Split remaining into train/valid
            valid_size = int(len(remaining_doc_ids) * (1 - self.train_valid_ratio))
            valid_doc_ids = remaining_doc_ids[:valid_size]
            train_doc_ids = remaining_doc_ids[valid_size:]

            self.logger.info(f"  Train: {len(train_doc_ids):,}")
            self.logger.info(f"  Valid: {len(valid_doc_ids):,}")
            self.logger.info(f"  Test: {len(test_doc_ids):,}")

            # Create metadata records
            train_records = [self.create_doc_metadata_record(doc_id, meta_index) for doc_id in train_doc_ids]
            valid_records = [self.create_doc_metadata_record(doc_id, meta_index) for doc_id in valid_doc_ids]
            test_records = [self.create_doc_metadata_record(doc_id, meta_index) for doc_id in test_doc_ids]

            # Add source info to test records for global set
            for record in test_records:
                record['source_split'] = continent
                record['split_type'] = 'continents'
            global_test_records.extend(test_records)

            # Save individual splits
            self._save_split(train_dir, f"{continent}_train_ids", train_records)
            self._save_split(valid_dir, f"{continent}_valid_ids", valid_records)
            self._save_split(test_dir, f"{continent}_test_ids", test_records)

            # Clear memory
            del train_records, valid_records, test_records
            gc.collect()

        # Save global test set
        global_test_file = os.path.join(self.global_test_dir, "continents_global_test.pkl")
        with open(global_test_file, 'wb') as f:
            pickle.dump(global_test_records, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.logger.info(f"✅ Saved global test set: {len(global_test_records):,} records")
        self.logger.info(f"✅ Continent splits completed!")

    def _save_split(self, split_dir, split_name, records):
        """Save split data and create metadata files"""
        # Save records
        split_file = os.path.join(split_dir, f"{split_name}.pkl")
        with open(split_file, 'wb') as f:
            pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Create metadata file
        self._create_metadata_file(split_dir, split_name, len(records))

        # Create distribution file
        self._create_theme_distribution_file(split_dir, split_name, records)

    def _create_metadata_file(self, split_dir, split_name, doc_count):
        """Create metadata file for split"""
        metadata = {
            'split_name': split_name,
            'num_documents': doc_count,
            'creation_timestamp': datetime.now().isoformat(),
            'seed': self.seed,
            'created_by': 'ids_only_data_splitter_v2',
            'data_format': 'doc_ids_and_metadata_only',
            'approach': 'individual_splits_with_global_test',
            'note': 'Use load_content_for_ids() to retrieve actual document content'
        }

        metadata_file = os.path.join(split_dir, f"{split_name}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def create_novel_concept_splits_ids_only(self, meta_index):
        """Create novel concept splits with individual train/valid/test + global test set"""
        self.logger.info("🔮 Creating novel concept splits ...")

        pivot_years = self.config.PIVOT_YEARS
        num_pivots = len(pivot_years)
        test_size_per_pivot = self.global_test_size // num_pivots

        self.logger.info(f"Pivot years: {pivot_years}")
        self.logger.info(f"Test size per pivot: {test_size_per_pivot}")

        global_test_records = []

        for pivot_year in pivot_years:
            self.logger.info(f"Processing pivot {pivot_year}...")

            # Create directories
            pivot_dir = os.path.join(self.individual_splits_dir, "novel_concept", f"pivot_{pivot_year}")
            train_dir = os.path.join(pivot_dir, "train")
            valid_dir = os.path.join(pivot_dir, "valid")
            test_dir = os.path.join(pivot_dir, "test")
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(valid_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            # Get all document IDs for this pivot (pre-pivot data)
            pre_pivot_ids = []
            for doc_id, doc_info in meta_index['documents'].items():
                if doc_info['year'] < pivot_year:
                    pre_pivot_ids.append(doc_id)

            random.shuffle(pre_pivot_ids)

            # Split: test first, then train/valid from remainder
            test_doc_ids = pre_pivot_ids[:test_size_per_pivot]
            remaining_doc_ids = pre_pivot_ids[test_size_per_pivot:]

            # Split remaining into train/valid
            valid_size = int(len(remaining_doc_ids) * (1 - self.train_valid_ratio))
            valid_doc_ids = remaining_doc_ids[:valid_size]
            train_doc_ids = remaining_doc_ids[valid_size:]

            self.logger.info(f"  Train: {len(train_doc_ids):,}")
            self.logger.info(f"  Valid: {len(valid_doc_ids):,}")
            self.logger.info(f"  Test: {len(test_doc_ids):,}")

            # Create metadata records
            train_records = [self.create_doc_metadata_record(doc_id, meta_index) for doc_id in train_doc_ids]
            valid_records = [self.create_doc_metadata_record(doc_id, meta_index) for doc_id in valid_doc_ids]
            test_records = [self.create_doc_metadata_record(doc_id, meta_index) for doc_id in test_doc_ids]

            # Add source info to test records for global set
            for record in test_records:
                record['source_split'] = f"pivot_{pivot_year}"
                record['split_type'] = 'novel-concept'
            global_test_records.extend(test_records)

            # Save individual splits
            self._save_split(train_dir, f"pre_{pivot_year}_train_ids", train_records)
            self._save_split(valid_dir, f"pre_{pivot_year}_valid_ids", valid_records)
            self._save_split(test_dir, f"pre_{pivot_year}_test_ids", test_records)

            # Clear memory
            del train_records, valid_records, test_records
            gc.collect()

        # Save global test set
        global_test_file = os.path.join(self.global_test_dir, "novel_concept_global_test.pkl")
        with open(global_test_file, 'wb') as f:
            pickle.dump(global_test_records, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.logger.info(f"✅ Saved global test set: {len(global_test_records):,} records")
        self.logger.info(f"✅ Novel concept splits completed!")

    def create_concept_change_splits_ids_only(self, meta_index):
        """Create concept change splits with individual train/valid/test + global test set"""
        self.logger.info("🎯 Creating concept change splits ...")

        # Get available themes
        available_themes = list(meta_index['by_theme'].keys())
        if not available_themes:
            self.logger.warning("No themes found in meta-index. Cannot create concept change splits.")
            return

        num_themes = len(available_themes)
        test_size_per_theme = self.global_test_size // num_themes

        self.logger.info(f"Themes: {available_themes}")
        self.logger.info(f"Test size per theme: {test_size_per_theme}")

        global_test_records = []

        for theme in available_themes:
            self.logger.info(f"Processing theme: {theme}...")

            # Create theme directory (sanitize theme name for filesystem)
            theme_key = theme.lower().replace(' ', '_').replace('/', '_').replace('\\', '_')
            theme_dir = os.path.join(self.individual_splits_dir, "concept_change", theme_key)
            train_dir = os.path.join(theme_dir, "train")
            valid_dir = os.path.join(theme_dir, "valid")
            test_dir = os.path.join(theme_dir, "test")
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(valid_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            # Get all document IDs from this theme
            all_doc_ids = meta_index['by_theme'][theme]
            random.shuffle(all_doc_ids)

            # Split: test first, then train/valid from remainder
            test_doc_ids = all_doc_ids[:test_size_per_theme]
            remaining_doc_ids = all_doc_ids[test_size_per_theme:]

            # Split remaining into train/valid
            valid_size = int(len(remaining_doc_ids) * (1 - self.train_valid_ratio))
            valid_doc_ids = remaining_doc_ids[:valid_size]
            train_doc_ids = remaining_doc_ids[valid_size:]

            self.logger.info(f"  Train: {len(train_doc_ids):,}")
            self.logger.info(f"  Valid: {len(valid_doc_ids):,}")
            self.logger.info(f"  Test: {len(test_doc_ids):,}")

            # Create metadata records
            train_records = [self.create_doc_metadata_record(doc_id, meta_index) for doc_id in train_doc_ids]
            valid_records = [self.create_doc_metadata_record(doc_id, meta_index) for doc_id in valid_doc_ids]
            test_records = [self.create_doc_metadata_record(doc_id, meta_index) for doc_id in test_doc_ids]

            # Add source info to test records for global set
            for record in test_records:
                record['source_split'] = theme
                record['split_type'] = 'concept-change'
            global_test_records.extend(test_records)

            # Save individual splits
            self._save_split(train_dir, f"{theme_key}_train_ids", train_records)
            self._save_split(valid_dir, f"{theme_key}_valid_ids", valid_records)
            self._save_split(test_dir, f"{theme_key}_test_ids", test_records)

            # Clear memory
            del train_records, valid_records, test_records
            gc.collect()

        # Save global test set
        global_test_file = os.path.join(self.global_test_dir, "concept_change_global_test.pkl")
        with open(global_test_file, 'wb') as f:
            pickle.dump(global_test_records, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.logger.info(f"✅ Saved global test set: {len(global_test_records):,} records")
        self.logger.info(f"✅ Concept change splits completed!")

    def load_content_for_ids(self, ids_file, output_file=None):
        """Utility function to load actual content for a set of document IDs"""
        self.logger.info(f"Loading content for IDs from {ids_file}...")

        # Load the ID records
        with open(ids_file, 'rb') as f:
            id_records = pickle.load(f)

        self.logger.info(f"Found {len(id_records):,} document records")

        # Group by file to minimize I/O
        files_to_docs = defaultdict(list)
        for record in id_records:
            files_to_docs[record['file_path']].append(record)

        # Load actual documents
        full_documents = []

        for file_path, doc_requests in files_to_docs.items():
            try:
                with open(file_path, 'rb') as f:
                    documents = pickle.load(f)

                # Extract requested documents
                for req in doc_requests:
                    if req['file_index'] < len(documents):
                        doc = documents[req['file_index']].copy()

                        # Add metadata
                        doc['year'] = req['year']
                        doc['country'] = req['country']
                        doc['continent'] = req['continent']

                        full_documents.append(doc)

                # Clear memory
                del documents

            except Exception as e:
                self.logger.error(f"Error loading documents from {file_path}: {e}")
                continue

        # Save full documents if output file specified
        if output_file:
            with open(output_file, 'wb') as f:
                pickle.dump(full_documents, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.logger.info(f"✅ Saved {len(full_documents):,} full documents to {output_file}")

        return full_documents

    def _create_theme_distribution_file(self, split_dir, split_name, records):
        """Create theme distribution analysis"""
        theme_dist = Counter()
        continent_dist = Counter()
        year_dist = Counter()

        for record in records:
            continent_dist[record['continent']] += 1
            year_dist[record['year']] += 1

            if record['dominant_theme']:
                theme_dist[record['dominant_theme']] += 1

        distribution = {
            'split_name': split_name,
            'total_documents': len(records),
            'theme_distribution': dict(theme_dist.most_common()),
            'continent_distribution': dict(continent_dist),
            'year_distribution': dict(year_dist),
            'documents_with_themes': sum(1 for r in records if r['dominant_theme']),
            'creation_timestamp': datetime.now().isoformat()
        }

        dist_file = os.path.join(split_dir, f"{split_name}_distribution.json")
        with open(dist_file, 'w') as f:
            json.dump(distribution, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="IDs-only data splitter with individual + global test sets")
    parser.add_argument("--split-type",
                       choices=['all', 'continents', 'novel-concept', 'concept-change'],
                       help="Type of split to create")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible splits")
    parser.add_argument("--train-valid-ratio", type=float, default=0.9,
                       help="Ratio of train vs valid after removing test (default: 0.9)")
    parser.add_argument("--global-test-size", type=int, default=10000,
                       help="Total size of global test set (default: 10000)")
    parser.add_argument("--load-content",
                       help="Load full content for an IDs file (provide path to *_ids.pkl file)")
    parser.add_argument("--output-content",
                       help="Output file for full content (use with --load-content)")
    parser.add_argument("--show-splits", action="store_true",
                       help="Show created splits status")

    args = parser.parse_args()

    # Create splitter
    splitter = IDsOnlyDataSplitter(
        seed=args.seed,
        train_valid_ratio=args.train_valid_ratio,
        global_test_size=args.global_test_size
    )

    if args.show_splits:
        # Show what splits have been created
        print("📋 CREATED SPLITS OVERVIEW")
        print("=" * 60)

        # Individual splits
        print("\n📁 INDIVIDUAL SPLITS:")
        individual_dir = splitter.individual_splits_dir
        if os.path.exists(individual_dir):
            for split_type in ['continents', 'novel_concept', 'concept_change']:
                split_type_dir = os.path.join(individual_dir, split_type)
                if os.path.exists(split_type_dir):
                    print(f"\n  {split_type.upper()}:")
                    for split_name in os.listdir(split_type_dir):
                        split_path = os.path.join(split_type_dir, split_name)
                        if os.path.isdir(split_path):
                            # Check for train/valid/test (handle different naming patterns)
                            if split_type == 'novel_concept':
                                # Novel concept uses pre_YEAR naming
                                year = split_name.replace('pivot_', '')
                                train_file = os.path.join(split_path, "train", f"pre_{year}_train_ids.pkl")
                                valid_file = os.path.join(split_path, "valid", f"pre_{year}_valid_ids.pkl")
                                test_file = os.path.join(split_path, "test", f"pre_{year}_test_ids.pkl")
                            else:
                                # Standard naming for continents and concept_change
                                train_file = os.path.join(split_path, "train", f"{split_name}_train_ids.pkl")
                                valid_file = os.path.join(split_path, "valid", f"{split_name}_valid_ids.pkl")
                                test_file = os.path.join(split_path, "test", f"{split_name}_test_ids.pkl")

                            if all(os.path.exists(f) for f in [train_file, valid_file, test_file]):
                                with open(train_file, 'rb') as f:
                                    train_size = len(pickle.load(f))
                                with open(valid_file, 'rb') as f:
                                    valid_size = len(pickle.load(f))
                                with open(test_file, 'rb') as f:
                                    test_size = len(pickle.load(f))

                                print(f"    {split_name}: {train_size:,} train, {valid_size:,} valid, {test_size:,} test")

        # Global test sets
        print("\n🌐 GLOBAL TEST SETS:")
        global_dir = splitter.global_test_dir
        if os.path.exists(global_dir):
            for global_file in os.listdir(global_dir):
                if global_file.endswith('_global_test.pkl'):
                    global_path = os.path.join(global_dir, global_file)
                    with open(global_path, 'rb') as f:
                        global_records = pickle.load(f)

                    # Count by source
                    source_counts = Counter(r['source_split'] for r in global_records)
                    split_type = global_file.replace('_global_test.pkl', '')

                    print(f"  {split_type}: {len(global_records):,} total")
                    for source, count in source_counts.items():
                        print(f"    - {source}: {count:,}")
        else:
            print("  ❌ No global test sets found")

        return

    if args.load_content:
        if not os.path.exists(args.load_content):
            print(f"❌ File not found: {args.load_content}")
            sys.exit(1)

        output_file = args.output_content or args.load_content.replace('_ids.pkl', '_full.pkl')

        try:
            documents = splitter.load_content_for_ids(args.load_content, output_file)
            print(f"✅ Loaded {len(documents):,} full documents")
            print(f"📁 Saved to: {output_file}")
        except Exception as e:
            print(f"❌ Error loading content: {e}")
            sys.exit(1)
        return

    if args.split_type:
        try:
            # Ensure meta-index exists
            if not os.path.exists(splitter.meta_index_file):
                print("❌ Meta-index not found. Run create_meta_index.py --create first.")
                sys.exit(1)

            print(f"🚀 Starting data splitting with seed {args.seed}")
            print(f"📋 Split type: {args.split_type}")
            print(f"📊 Global test size: {args.global_test_size}")
            print(f"📊 Train/Valid ratio: {args.train_valid_ratio}")

            # Load meta-index
            meta_index = splitter.load_meta_index()

            if args.split_type in ['all', 'continents']:
                print("\n🌍 Creating continent splits...")
                splitter.create_continent_splits_ids_only(meta_index)

            if args.split_type in ['all', 'novel-concept']:
                print("\n🔮 Creating novel concept splits...")
                splitter.create_novel_concept_splits_ids_only(meta_index)

            if args.split_type in ['all', 'concept-change']:
                print("\n🎯 Creating concept change splits...")
                splitter.create_concept_change_splits_ids_only(meta_index)

            print(f"\n✅ Data splitting completed successfully!")
            print(f"📁 Individual splits: {splitter.individual_splits_dir}")
            print(f"📁 Global test sets: {splitter.global_test_dir}")
            print(f"💡 Use --show-splits to see what was created")
            print(f"💡 Use --load-content to retrieve full document content when needed")

        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
