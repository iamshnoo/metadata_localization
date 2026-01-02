#!/usr/bin/env python3
"""
STREAMING HuggingFace dataset creator that handles ALL split types.

This version:
1. Supports continent splits, novel concept splits, and concept change splits
2. Uses streaming approach to avoid memory issues
3. Processes documents one at a time using generators
4. Perfect for large datasets (4M+ documents)

Usage:
    # Continent splits
    python create_hf_datasets_streaming.py --split-type continents --split-name africa --train-only
    python create_hf_datasets_streaming.py --split-type continents --split-name africa
    python create_hf_datasets_streaming.py --split-type continents
    
    # Novel concept splits
    python create_hf_datasets_streaming.py --split-type novel-concept --split-name pivot_2018
    python create_hf_datasets_streaming.py --split-type novel-concept
    
    # Concept change splits
    python create_hf_datasets_streaming.py --split-type concept-change --split-name politics
    python create_hf_datasets_streaming.py --split-type concept-change
    
    # All splits
    python create_hf_datasets_streaming.py --split-type all
"""

import sys
import argparse
import logging
import os
import pickle
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from tqdm import tqdm
import gc
import time

# HuggingFace imports
from datasets import Dataset, DatasetDict, Features, Value, Sequence
import pandas as pd

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import Config

class StreamingDatasetCreator:
    def __init__(self):
        self.config = Config()
        
        # Setup logging
        self.setup_logging()
        
        # Output directory for HF datasets
        self.hf_datasets_dir = os.path.join(self.config.TRAINING_DATA_BASE, "hf_datasets")
        os.makedirs(self.hf_datasets_dir, exist_ok=True)
        
        # Define the schema for our dataset
        self.features = Features({
            'doc_id': Value('string'),
            'url': Value('string'),
            'country': Value('string'),
            'continent': Value('string'),
            'year': Value('int32'),
            'title': Value('string'),
            'content': Value('string'),
            'topics': Sequence(Value('string')),
            'topic_probabilities': Sequence(Value('float32')),
            'num_topics': Value('int32'),
            'theme': Value('string'),
            'has_themes': Value('bool')
        })
        
        self.logger.info(f"Initialized STREAMING HuggingFace dataset creator")
        self.logger.info(f"Output directory: {self.hf_datasets_dir}")

    def setup_logging(self):
        """Setup logging"""
        log_dir = os.path.join(self.config.BASE, "metacul/logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, "create_hf_datasets_streaming.log")
        
        # Create logger
        self.logger = logging.getLogger("create_hf_datasets_streaming")
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

    def document_generator(self, id_records, split_name):
        """Generator that yields documents one at a time"""
        self.logger.info(f"🚀 Creating document generator for {len(id_records):,} documents...")
        
        # Group by file for efficient loading
        files_to_docs = defaultdict(list)
        for i, record in enumerate(id_records):
            files_to_docs[record['file_path']].append((i, record))
        
        # Sort files by size (largest first)
        sorted_files = sorted(files_to_docs.items(), key=lambda x: len(x[1]), reverse=True)
        
        self.logger.info(f"📊 Will process {len(sorted_files)} files")
        
        docs_yielded = 0
        
        # Process files one by one
        with tqdm(total=len(sorted_files), desc=f"Processing files for {split_name}") as file_pbar:
            for file_path, doc_requests in sorted_files:
                try:
                    # Load file
                    with open(file_path, 'rb') as f:
                        documents = pickle.load(f)
                    
                    # Process documents from this file
                    for original_idx, req in doc_requests:
                        if req['file_index'] < len(documents):
                            doc = documents[req['file_index']]
                            
                            # Extract topics
                            topics = doc.get('topics', [])
                            topic_labels = []
                            topic_probs = []
                            
                            if topics:
                                sorted_topics = sorted(topics, key=lambda x: x.get('probability', 0), reverse=True)
                                for topic in sorted_topics:
                                    topic_labels.append(topic.get('topic_label', 'Unknown'))
                                    topic_probs.append(float(topic.get('probability', 0.0)))
                            
                            # Create record in your exact format
                            record = {
                                'doc_id': str(doc.get('article_id', '')),
                                'url': str(doc.get('url', '')),
                                'country': str(req['country']),
                                'continent': str(req['continent']),
                                'year': int(req['year']),
                                'title': str(doc.get('title', '')),
                                'content': str(doc.get('content', '')),
                                'topics': topic_labels,
                                'topic_probabilities': topic_probs,
                                'num_topics': len(topics),
                                'theme': str(req['dominant_theme'] if req['dominant_theme'] else ''),
                                'has_themes': bool(req['has_themes'])
                            }
                            
                            yield record
                            docs_yielded += 1
                    
                    # Clear memory
                    del documents
                    
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {e}")
                
                # Update progress
                file_pbar.set_postfix({
                    'docs': f"{docs_yielded:,}",
                    'file': os.path.basename(file_path)[:20]
                })
                file_pbar.update(1)
                
                # Periodic garbage collection
                if file_pbar.n % 10 == 0:
                    gc.collect()
        
        self.logger.info(f"✅ Generator complete: {docs_yielded:,} documents yielded")

    def create_streaming_dataset(self, id_records, split_name):
        """Create dataset using streaming approach"""
        self.logger.info(f"📦 Creating streaming dataset for {split_name}...")
        
        start_time = time.time()
        
        # Create dataset from generator
        dataset = Dataset.from_generator(
            lambda: self.document_generator(id_records, split_name),
            features=self.features,
            num_proc=1  # Single process to avoid memory issues
        )
        
        total_time = time.time() - start_time
        
        self.logger.info(f"✅ Created streaming dataset:")
        self.logger.info(f"   Records: {len(dataset):,}")
        self.logger.info(f"   Time: {total_time:.1f}s")
        self.logger.info(f"   Features: {list(dataset.features.keys())}")
        
        # Show sample record
        if len(dataset) > 0:
            sample = dataset[0]
            self.logger.info(f"📋 Sample record:")
            self.logger.info(f"   doc_id: {sample['doc_id']}")
            self.logger.info(f"   country: {sample['country']}")
            self.logger.info(f"   continent: {sample['continent']}")
            self.logger.info(f"   year: {sample['year']}")
            self.logger.info(f"   title: {sample['title'][:50]}...")
            self.logger.info(f"   topics: {len(sample['topics'])} topics")
            self.logger.info(f"   theme: {sample['theme']}")
        
        return dataset

    def process_continent_splits(self, specific_continent=None, train_only=False):
        """Process continent-based splits"""
        mode_str = "TRAIN-ONLY" if train_only else "FULL"
        self.logger.info(f"🌍 Processing continent splits ({mode_str} - STREAMING)...")
        
        continents_dir = os.path.join(self.config.TRAINING_DATA_BASE, "individual_splits", "continents")
        
        if not os.path.exists(continents_dir):
            self.logger.warning("No continent splits found")
            return
        
        # Get continents
        continents = [item for item in os.listdir(continents_dir) 
                     if os.path.isdir(os.path.join(continents_dir, item))]
        
        if specific_continent:
            continents = [c for c in continents if c.lower() == specific_continent.lower()]
            if not continents:
                self.logger.error(f"Continent '{specific_continent}' not found")
                return
        
        self.logger.info(f"📋 Found continents: {continents}")
        
        for continent_idx, continent in enumerate(continents):
            self.logger.info(f"🌍 Processing {continent} ({continent_idx+1}/{len(continents)})")
            self.logger.info("=" * 60)
            
            continent_dir = os.path.join(continents_dir, continent)
            train_file = os.path.join(continent_dir, "train", f"{continent}_train_ids.pkl")
            valid_file = os.path.join(continent_dir, "valid", f"{continent}_valid_ids.pkl")
            test_file = os.path.join(continent_dir, "test", f"{continent}_test_ids.pkl")
            
            # Check files exist
            if not os.path.exists(train_file):
                self.logger.warning(f"Missing train file for {continent}")
                continue
            
            if not train_only and not os.path.exists(valid_file):
                self.logger.warning(f"Missing valid file for {continent}")
                continue
            
            if not train_only and not os.path.exists(test_file):
                self.logger.warning(f"Missing test file for {continent}")
                continue
            
            # Load records
            self.logger.info("📂 Loading ID records...")
            with open(train_file, 'rb') as f:
                train_records = pickle.load(f)
            
            self.logger.info(f"   Train records: {len(train_records):,}")
            
            if not train_only:
                with open(valid_file, 'rb') as f:
                    valid_records = pickle.load(f)
                self.logger.info(f"   Valid records: {len(valid_records):,}")
                
                with open(test_file, 'rb') as f:
                    test_records = pickle.load(f)
                self.logger.info(f"   Test records: {len(test_records):,}")
            
            # Create datasets using streaming
            self.logger.info("🚂 Creating TRAIN dataset (streaming)...")
            train_dataset = self.create_streaming_dataset(train_records, f"{continent}_train")
            
            if train_only:
                # Save train-only dataset
                dataset_dict = DatasetDict({'train': train_dataset})
                output_path = os.path.join(self.hf_datasets_dir, "continents", f"{continent}_train_only")
            else:
                # Create valid dataset
                self.logger.info("✅ Creating VALID dataset (streaming)...")
                valid_dataset = self.create_streaming_dataset(valid_records, f"{continent}_valid")
                
                # Create test dataset
                self.logger.info("🧪 Creating TEST dataset (streaming)...")
                test_dataset = self.create_streaming_dataset(test_records, f"{continent}_test")
                
                # Create full dataset with 3 splits
                dataset_dict = DatasetDict({
                    'train': train_dataset,
                    'validation': valid_dataset,
                    'test': test_dataset
                })
                output_path = os.path.join(self.hf_datasets_dir, "continents", continent)
            
            # Save dataset
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            self.logger.info(f"💾 Saving dataset to {output_path}...")
            dataset_dict.save_to_disk(output_path)
            
            self.logger.info(f"✅ Completed {continent}!")
            self.logger.info(f"   Train: {len(train_dataset):,} samples")
            if not train_only:
                self.logger.info(f"   Valid: {len(valid_dataset):,} samples")
                self.logger.info(f"   Test: {len(test_dataset):,} samples")
            
            # Cleanup
            del train_records, train_dataset
            if not train_only:
                del valid_records, valid_dataset, test_records, test_dataset
            del dataset_dict
            gc.collect()
            
            self.logger.info(f"🎯 {continent} complete!")
            self.logger.info("-" * 60)

    def process_novel_concept_splits(self, specific_pivot=None):
        """Process novel concept splits (temporal analysis)"""
        self.logger.info("🔮 Processing novel concept splits (STREAMING)...")
        
        novel_concept_dir = os.path.join(self.config.TRAINING_DATA_BASE, "individual_splits", "novel_concept")
        
        if not os.path.exists(novel_concept_dir):
            self.logger.warning("No novel concept splits found")
            return
        
        # Get pivot years
        pivot_years = self.config.PIVOT_YEARS
        
        # Filter to specific pivot if requested
        if specific_pivot:
            try:
                pivot_year = int(specific_pivot.replace('pivot_', ''))
                pivot_years = [pivot_year] if pivot_year in pivot_years else []
                if not pivot_years:
                    self.logger.error(f"Pivot year '{pivot_year}' not found in config")
                    return
            except ValueError:
                self.logger.error(f"Invalid pivot format: {specific_pivot}")
                return
        
        self.logger.info(f"📋 Found pivot years: {pivot_years}")
        
        for pivot_idx, pivot_year in enumerate(pivot_years):
            self.logger.info(f"🔮 Processing pivot_{pivot_year} ({pivot_idx+1}/{len(pivot_years)})")
            self.logger.info("=" * 60)
            
            pivot_dir = os.path.join(novel_concept_dir, f"pivot_{pivot_year}")
            train_file = os.path.join(pivot_dir, "train", f"pre_{pivot_year}_train_ids.pkl")
            valid_file = os.path.join(pivot_dir, "valid", f"pre_{pivot_year}_valid_ids.pkl")
            test_file = os.path.join(pivot_dir, "test", f"pre_{pivot_year}_test_ids.pkl")
            
            if not (os.path.exists(train_file) and os.path.exists(valid_file) and os.path.exists(test_file)):
                self.logger.warning(f"Missing files for pivot {pivot_year}")
                continue
            
            # Load records
            self.logger.info("📂 Loading ID records...")
            with open(train_file, 'rb') as f:
                train_records = pickle.load(f)
            with open(valid_file, 'rb') as f:
                valid_records = pickle.load(f)
            with open(test_file, 'rb') as f:
                test_records = pickle.load(f)
            
            self.logger.info(f"   Pre-{pivot_year} (train): {len(train_records):,}")
            self.logger.info(f"   Pre-{pivot_year} (valid): {len(valid_records):,}")
            self.logger.info(f"   Pre-{pivot_year} (test): {len(test_records):,}")
            
            # Create datasets using streaming
            self.logger.info(f"🚂 Creating PRE-{pivot_year} TRAIN dataset (streaming)...")
            train_dataset = self.create_streaming_dataset(train_records, f"pre_{pivot_year}_train")
            
            self.logger.info(f"✅ Creating PRE-{pivot_year} VALID dataset (streaming)...")
            valid_dataset = self.create_streaming_dataset(valid_records, f"pre_{pivot_year}_valid")
            
            self.logger.info(f"🧪 Creating PRE-{pivot_year} TEST dataset (streaming)...")
            test_dataset = self.create_streaming_dataset(test_records, f"pre_{pivot_year}_test")
            
            # Create dataset dict with 3 splits
            dataset_dict = DatasetDict({
                'train': train_dataset,
                'validation': valid_dataset,
                'test': test_dataset
            })
            
            # Save dataset
            output_path = os.path.join(self.hf_datasets_dir, "novel_concept", f"pivot_{pivot_year}")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            self.logger.info(f"💾 Saving dataset to {output_path}...")
            dataset_dict.save_to_disk(output_path)
            
            self.logger.info(f"✅ Completed pivot_{pivot_year}!")
            self.logger.info(f"   Train (pre-{pivot_year}): {len(train_dataset):,} samples")
            self.logger.info(f"   Valid (pre-{pivot_year}): {len(valid_dataset):,} samples")
            self.logger.info(f"   Test (pre-{pivot_year}): {len(test_dataset):,} samples")
            
            # Cleanup
            del train_records, valid_records, test_records
            del train_dataset, valid_dataset, test_dataset, dataset_dict
            gc.collect()
            
            self.logger.info(f"🎯 pivot_{pivot_year} complete!")
            self.logger.info("-" * 60)

    def process_concept_change_splits(self, specific_theme=None):
        """Process concept change splits (theme classification)"""
        self.logger.info("🎯 Processing concept change splits (STREAMING)...")
        
        concept_change_dir = os.path.join(self.config.TRAINING_DATA_BASE, "individual_splits", "concept_change")
        
        if not os.path.exists(concept_change_dir):
            self.logger.warning("No concept change splits found")
            return
        
        # Get list of themes
        themes = []
        for item in os.listdir(concept_change_dir):
            theme_path = os.path.join(concept_change_dir, item)
            if os.path.isdir(theme_path):
                themes.append(item)
        
        # Filter to specific theme if requested
        if specific_theme:
            theme_key = specific_theme.lower().replace(' ', '_')
            themes = [t for t in themes if t.lower() == theme_key]
            if not themes:
                self.logger.error(f"Theme '{specific_theme}' not found")
                return
        
        self.logger.info(f"📋 Found themes: {themes}")
        
        for theme_idx, theme in enumerate(themes):
            self.logger.info(f"🎯 Processing {theme} ({theme_idx+1}/{len(themes)})")
            self.logger.info("=" * 60)
            
            theme_dir = os.path.join(concept_change_dir, theme)
            train_file = os.path.join(theme_dir, "train", f"{theme}_train_ids.pkl")
            valid_file = os.path.join(theme_dir, "valid", f"{theme}_valid_ids.pkl")
            test_file = os.path.join(theme_dir, "test", f"{theme}_test_ids.pkl")
            
            if not (os.path.exists(train_file) and os.path.exists(valid_file) and os.path.exists(test_file)):
                self.logger.warning(f"Missing files for theme {theme}")
                continue
            
            # Load records
            self.logger.info("📂 Loading ID records...")
            with open(train_file, 'rb') as f:
                train_records = pickle.load(f)
            with open(valid_file, 'rb') as f:
                valid_records = pickle.load(f)
            with open(test_file, 'rb') as f:
                test_records = pickle.load(f)
            
            self.logger.info(f"   Train: {len(train_records):,}")
            self.logger.info(f"   Valid: {len(valid_records):,}")
            self.logger.info(f"   Test: {len(test_records):,}")
            
            # Create datasets using streaming
            self.logger.info(f"🚂 Creating {theme} TRAIN dataset (streaming)...")
            train_dataset = self.create_streaming_dataset(train_records, f"{theme}_train")
            
            self.logger.info(f"✅ Creating {theme} VALID dataset (streaming)...")
            valid_dataset = self.create_streaming_dataset(valid_records, f"{theme}_valid")
            
            self.logger.info(f"🧪 Creating {theme} TEST dataset (streaming)...")
            test_dataset = self.create_streaming_dataset(test_records, f"{theme}_test")
            
            # Create dataset dict with 3 splits
            dataset_dict = DatasetDict({
                'train': train_dataset,
                'validation': valid_dataset,
                'test': test_dataset
            })
            
            # Save dataset
            output_path = os.path.join(self.hf_datasets_dir, "concept_change", theme)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            self.logger.info(f"💾 Saving dataset to {output_path}...")
            dataset_dict.save_to_disk(output_path)
            
            self.logger.info(f"✅ Completed {theme}!")
            self.logger.info(f"   Train: {len(train_dataset):,} samples")
            self.logger.info(f"   Valid: {len(valid_dataset):,} samples")
            self.logger.info(f"   Test: {len(test_dataset):,} samples")
            
            # Cleanup
            del train_records, valid_records, test_records
            del train_dataset, valid_dataset, test_dataset, dataset_dict
            gc.collect()
            
            self.logger.info(f"🎯 {theme} complete!")
            self.logger.info("-" * 60)

    def show_dataset_summary(self):
        """Show summary of all created datasets"""
        self.logger.info("📊 DATASET SUMMARY")
        self.logger.info("=" * 60)
        
        if not os.path.exists(self.hf_datasets_dir):
            self.logger.info("❌ No datasets found")
            return
        
        total_datasets = 0
        total_samples = 0
        
        # Check continent datasets
        continents_dir = os.path.join(self.hf_datasets_dir, "continents")
        if os.path.exists(continents_dir):
            self.logger.info("\n🌍 CONTINENT DATASETS:")
            for item in sorted(os.listdir(continents_dir)):
                dataset_path = os.path.join(continents_dir, item)
                if os.path.isdir(dataset_path):
                    try:
                        dataset_dict = DatasetDict.load_from_disk(dataset_path)
                        splits = list(dataset_dict.keys())
                        
                        if 'train' in splits and 'test' in splits:
                            train_size = len(dataset_dict['train'])
                            test_size = len(dataset_dict['test'])
                            if 'validation' in splits:
                                valid_size = len(dataset_dict['validation'])
                                total_samples += train_size + valid_size + test_size
                                self.logger.info(f"  ✅ {item}: {train_size:,} train, {valid_size:,} valid, {test_size:,} test")
                            else:
                                total_samples += train_size + test_size
                                self.logger.info(f"  ✅ {item}: {train_size:,} train, {test_size:,} test")
                        elif 'train' in splits:
                            train_size = len(dataset_dict['train'])
                            total_samples += train_size
                            self.logger.info(f"  ✅ {item}: {train_size:,} train only")
                        
                        total_datasets += 1
                        
                    except Exception as e:
                        self.logger.info(f"  ❌ {item}: Error loading - {e}")
        
        # Check novel concept datasets
        novel_dir = os.path.join(self.hf_datasets_dir, "novel_concept")
        if os.path.exists(novel_dir):
            self.logger.info("\n🔮 NOVEL CONCEPT DATASETS:")
            for item in sorted(os.listdir(novel_dir)):
                dataset_path = os.path.join(novel_dir, item)
                if os.path.isdir(dataset_path):
                    try:
                        dataset_dict = DatasetDict.load_from_disk(dataset_path)
                        train_size = len(dataset_dict['train'])
                        valid_size = len(dataset_dict['validation'])
                        test_size = len(dataset_dict['test'])
                        total_samples += train_size + valid_size + test_size
                        total_datasets += 1
                        
                        year = item.split('_')[1]
                        self.logger.info(f"  ✅ {item}: {train_size:,} train, {valid_size:,} valid, {test_size:,} test (pre-{year})")
                        
                    except Exception as e:
                        self.logger.info(f"  ❌ {item}: Error loading - {e}")
        
        # Check concept change datasets
        concept_dir = os.path.join(self.hf_datasets_dir, "concept_change")
        if os.path.exists(concept_dir):
            self.logger.info("\n🎯 CONCEPT CHANGE DATASETS:")
            for item in sorted(os.listdir(concept_dir)):
                dataset_path = os.path.join(concept_dir, item)
                if os.path.isdir(dataset_path):
                    try:
                        dataset_dict = DatasetDict.load_from_disk(dataset_path)
                        train_size = len(dataset_dict['train'])
                        valid_size = len(dataset_dict['validation'])
                        test_size = len(dataset_dict['test'])
                        total_samples += train_size + valid_size + test_size
                        total_datasets += 1
                        
                        self.logger.info(f"  ✅ {item}: {train_size:,} train, {valid_size:,} valid, {test_size:,} test")
                        
                    except Exception as e:
                        self.logger.info(f"  ❌ {item}: Error loading - {e}")
        
        self.logger.info(f"\n📈 TOTALS:")
        self.logger.info(f"  Total datasets: {total_datasets}")
        self.logger.info(f"  Total samples: {total_samples:,}")
        
        self.logger.info(f"\n🚀 USAGE EXAMPLES:")
        self.logger.info(f"  from datasets import DatasetDict")
        self.logger.info(f"  ")
        self.logger.info(f"  # Load continent dataset")
        self.logger.info(f"  dataset = DatasetDict.load_from_disk('training_data/hf_datasets/continents/africa')")
        self.logger.info(f"  ")
        self.logger.info(f"  # Load novel concept dataset")
        self.logger.info(f"  dataset = DatasetDict.load_from_disk('training_data/hf_datasets/novel_concept/pivot_2018')")
        self.logger.info(f"  ")
        self.logger.info(f"  # Load concept change dataset")
        self.logger.info(f"  dataset = DatasetDict.load_from_disk('training_data/hf_datasets/concept_change/politics')")

def main():
    parser = argparse.ArgumentParser(description="Create HuggingFace datasets (STREAMING - ALL SPLITS)")
    parser.add_argument("--split-type", choices=["all", "continents", "novel-concept", "concept-change"], 
                       default="continents", help="Type of splits to process")
    parser.add_argument("--split-name", help="Specific split name to process")
    parser.add_argument("--train-only", action="store_true", help="Process only train splits (continents only)")
    parser.add_argument("--show-summary", action="store_true", help="Show summary of created datasets")
    
    args = parser.parse_args()
    
    creator = StreamingDatasetCreator()
    
    if args.show_summary:
        creator.show_dataset_summary()
        return
    
    try:
        # Process based on split type
        if args.split_type in ["all", "continents"]:
            creator.process_continent_splits(args.split_name, args.train_only)
        
        if args.split_type in ["all", "novel-concept"]:
            creator.process_novel_concept_splits(args.split_name)
        
        if args.split_type in ["all", "concept-change"]:
            creator.process_concept_change_splits(args.split_name)
        
        # Show final summary
        creator.show_dataset_summary()
        
    except Exception as e:
        creator.logger.error(f"Fatal error: {e}")
        import traceback
        creator.logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
