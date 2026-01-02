#!/usr/bin/env python3
"""
Create meta-metadata index for document splitting.

This creates a lightweight index mapping document IDs to their metadata and file locations.
The index is used by data_splitter_ids_only.py for fast splitting operations.

Usage:
    python create_meta_index.py --create
    python create_meta_index.py --show-status
    python create_meta_index.py --clear
"""

import sys
import argparse
import logging
import os
import pickle
import json
import glob
import gc
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from tqdm import tqdm

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import Config

class MetaIndexCreator:
    def __init__(self, seed=42):
        self.config = Config()
        self.seed = seed
        
        # Setup logging
        self.setup_logging()
        
        # Cache directory
        self.cache_dir = os.path.join(self.config.TRAINING_DATA_BASE, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Meta-index file
        self.meta_index_file = os.path.join(self.cache_dir, f"meta_index_{seed}.pkl")
        
        self.logger.info(f"Initialized meta-index creator with seed {seed}")

    def setup_logging(self):
        """Setup logging"""
        log_dir = os.path.join(self.config.BASE, "metacul/logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, "create_meta_index.log")
        
        # Create logger
        self.logger = logging.getLogger("create_meta_index")
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

    def create_meta_index(self, force=False):
        """Create lightweight meta-metadata index"""
        self.logger.info("📋 Creating meta-metadata index...")
        
        # Check if index already exists
        if os.path.exists(self.meta_index_file) and not force:
            self.logger.info("Meta-index already exists. Use --force to recreate.")
            return self.load_meta_index()
        
        # Find all document metadata files
        metadata_pattern = os.path.join(
            self.config.BASE,
            "metacul/document_metadata/*/document_metadata_*.pkl"
        )
        metadata_files = glob.glob(metadata_pattern)
        
        if not metadata_files:
            raise FileNotFoundError(f"No document metadata files found at: {metadata_pattern}")
        
        self.logger.info(f"Found {len(metadata_files)} metadata files")
        
        # Initialize meta-index
        meta_index = {
            'documents': {},  # doc_id -> {file_path, file_index, year, country, continent, has_themes, dominant_theme}
            'by_continent': defaultdict(list),  # continent -> [doc_ids]
            'by_year': defaultdict(list),  # year -> [doc_ids]
            'by_theme': defaultdict(list),  # theme -> [doc_ids]
            'stats': {
                'total_documents': 0,
                'continent_counts': defaultdict(int),
                'year_counts': defaultdict(int),
                'theme_counts': defaultdict(int),
                'files_processed': 0,
                'files_failed': 0
            },
            'generation_timestamp': datetime.now().isoformat(),
            'seed': self.seed,
            'created_by': 'create_meta_index'
        }
        
        # Process each file to build index
        for metadata_file in tqdm(metadata_files, desc="📁 Indexing files"):
            try:
                # Extract year and country from filename
                filename = os.path.basename(metadata_file)
                parts = filename.replace('document_metadata_', '').replace('.pkl', '').split('_')
                
                if len(parts) >= 2:
                    year, country = int(parts[0]), parts[1]
                else:
                    self.logger.warning(f"Could not parse year/country from {filename}")
                    continue
                
                continent = self.config.COUNTRY_TO_CONTINENT.get(country)
                if not continent:
                    self.logger.warning(f"Unknown continent for country {country}")
                    continue
                
                # Load documents to build index (but don't keep them in memory)
                try:
                    with open(metadata_file, 'rb') as f:
                        documents = pickle.load(f)
                    
                    # Index each document
                    for doc_index, doc in enumerate(documents):
                        # Validate required fields
                        required_fields = ['article_id', 'title', 'content']
                        if not all(field in doc and doc[field] for field in required_fields):
                            continue
                        
                        doc_id = doc['article_id']
                        
                        # Check for themes
                        has_themes = False
                        dominant_theme = None
                        if 'themes' in doc:
                            themes = doc.get('themes', {})
                            dominant_theme_info = themes.get('dominant_theme')
                            if dominant_theme_info:
                                has_themes = True
                                dominant_theme = dominant_theme_info.get('theme_name')
                        
                        # Store document metadata
                        meta_index['documents'][doc_id] = {
                            'file_path': metadata_file,
                            'file_index': doc_index,
                            'year': year,
                            'country': country,
                            'continent': continent,
                            'has_themes': has_themes,
                            'dominant_theme': dominant_theme
                        }
                        
                        # Add to indices
                        meta_index['by_continent'][continent].append(doc_id)
                        meta_index['by_year'][year].append(doc_id)
                        
                        if dominant_theme:
                            meta_index['by_theme'][dominant_theme].append(doc_id)
                        
                        # Update stats
                        meta_index['stats']['total_documents'] += 1
                        meta_index['stats']['continent_counts'][continent] += 1
                        meta_index['stats']['year_counts'][year] += 1
                        if dominant_theme:
                            meta_index['stats']['theme_counts'][dominant_theme] += 1
                    
                    meta_index['stats']['files_processed'] += 1
                    
                    # Clear memory immediately
                    del documents
                    gc.collect()
                    
                except Exception as load_error:
                    self.logger.error(f"Failed to load {metadata_file}: {load_error}")
                    meta_index['stats']['files_failed'] += 1
                    continue
                    
            except Exception as e:
                self.logger.error(f"Error processing {metadata_file}: {e}")
                meta_index['stats']['files_failed'] += 1
                continue
        
        # Convert defaultdicts to regular dicts for serialization
        meta_index['by_continent'] = dict(meta_index['by_continent'])
        meta_index['by_year'] = dict(meta_index['by_year'])
        meta_index['by_theme'] = dict(meta_index['by_theme'])
        meta_index['stats']['continent_counts'] = dict(meta_index['stats']['continent_counts'])
        meta_index['stats']['year_counts'] = dict(meta_index['stats']['year_counts'])
        meta_index['stats']['theme_counts'] = dict(meta_index['stats']['theme_counts'])
        
        # Save meta-index
        try:
            self.logger.info("💾 Saving meta-index...")
            with open(self.meta_index_file, 'wb') as f:
                pickle.dump(meta_index, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            file_size = os.path.getsize(self.meta_index_file) / (1024 * 1024)  # MB
            self.logger.info(f"✅ Saved meta-index ({file_size:.1f} MB)")
            
        except Exception as e:
            self.logger.error(f"Failed to save meta-index: {e}")
            raise
        
        # Print summary
        stats = meta_index['stats']
        self.logger.info(f"✅ Meta-index created successfully:")
        self.logger.info(f"  📄 Total documents: {stats['total_documents']:,}")
        self.logger.info(f"  📁 Files processed: {stats['files_processed']}/{stats['files_processed'] + stats['files_failed']}")
        self.logger.info(f"  🌍 Continents: {list(stats['continent_counts'].keys())}")
        self.logger.info(f"  🎯 Themes: {len(stats['theme_counts'])} found")
        
        return meta_index

    def load_meta_index(self):
        """Load existing meta-index"""
        if not os.path.exists(self.meta_index_file):
            raise FileNotFoundError(f"Meta-index not found: {self.meta_index_file}")
        
        try:
            with open(self.meta_index_file, 'rb') as f:
                meta_index = pickle.load(f)
            
            file_size = os.path.getsize(self.meta_index_file) / (1024 * 1024)  # MB
            self.logger.info(f"📂 Loaded meta-index ({file_size:.1f} MB)")
            
            return meta_index
            
        except Exception as e:
            self.logger.error(f"Failed to load meta-index: {e}")
            raise

    def show_status(self):
        """Show meta-index status"""
        if not os.path.exists(self.meta_index_file):
            print("📋 META-INDEX STATUS: ❌ Not created")
            print(f"Expected location: {self.meta_index_file}")
            return
        
        try:
            meta_index = self.load_meta_index()
            stats = meta_index['stats']
            
            file_size = os.path.getsize(self.meta_index_file) / (1024 * 1024)  # MB
            creation_time = datetime.fromisoformat(meta_index['generation_timestamp'])
            
            print("📋 META-INDEX STATUS")
            print("=" * 50)
            print(f"Status: ✅ Available")
            print(f"File: {os.path.basename(self.meta_index_file)}")
            print(f"Size: {file_size:.1f} MB")
            print(f"Created: {creation_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Seed: {meta_index['seed']}")
            
            print(f"\n📊 CONTENT SUMMARY:")
            print(f"Total documents: {stats['total_documents']:,}")
            print(f"Files processed: {stats['files_processed']}")
            print(f"Files failed: {stats['files_failed']}")
            
            print(f"\n🌍 CONTINENTS:")
            for continent, count in stats['continent_counts'].items():
                print(f"  {continent}: {count:,} documents")
            
            print(f"\n🎯 THEMES:")
            theme_items = sorted(stats['theme_counts'].items(), key=lambda x: x[1], reverse=True)
            for theme, count in theme_items[:10]:  # Top 10 themes
                print(f"  {theme}: {count:,} documents")
            
            if len(theme_items) > 10:
                print(f"  ... and {len(theme_items) - 10} more themes")
            
            print(f"\n📅 YEAR RANGE:")
            years = list(stats['year_counts'].keys())
            if years:
                print(f"  {min(years)} - {max(years)} ({len(years)} years)")
            
        except Exception as e:
            print(f"❌ Error reading meta-index: {e}")

    def clear_meta_index(self):
        """Clear/delete the meta-index"""
        if os.path.exists(self.meta_index_file):
            try:
                os.remove(self.meta_index_file)
                print(f"✅ Meta-index cleared: {os.path.basename(self.meta_index_file)}")
            except Exception as e:
                print(f"❌ Failed to clear meta-index: {e}")
        else:
            print("ℹ️  No meta-index to clear")

def main():
    parser = argparse.ArgumentParser(description="Create meta-metadata index for document splitting")
    parser.add_argument("--create", action="store_true",
                       help="Create meta-index")
    parser.add_argument("--force", action="store_true",
                       help="Force recreation of existing meta-index")
    parser.add_argument("--show-status", action="store_true",
                       help="Show meta-index status")
    parser.add_argument("--clear", action="store_true",
                       help="Clear/delete meta-index")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for meta-index (default: 42)")
    
    args = parser.parse_args()
    
    # Create meta-index creator
    creator = MetaIndexCreator(seed=args.seed)
    
    if args.show_status:
        creator.show_status()
        return
    
    if args.clear:
        creator.clear_meta_index()
        return
    
    if args.create:
        try:
            print(f"🚀 Creating meta-index with seed {args.seed}")
            if args.force:
                print("🔄 Force mode: Will overwrite existing meta-index")
            
            meta_index = creator.create_meta_index(force=args.force)
            
            print(f"\n✅ Meta-index creation completed successfully!")
            print(f"📁 Location: {creator.meta_index_file}")
            print(f"💡 Use --show-status to view details")
            print(f"💡 Use data_splitter_ids_only.py to create splits")
            
        except Exception as e:
            print(f"❌ Error creating meta-index: {e}")
            sys.exit(1)
    else:
        parser.print_help()
        print(f"\n💡 TYPICAL WORKFLOW:")
        print(f"  1. python create_meta_index.py --create")
        print(f"  2. python create_meta_index.py --show-status")
        print(f"  3. python data_splitter_ids_only.py --split-type all")

if __name__ == "__main__":
    main()
