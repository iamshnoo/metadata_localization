#!/usr/bin/env python3
"""
Update all document metadata files with theme information.

This script:
1. Loads the theme mapping from dominant_themes.json
2. Updates all document metadata files to include theme information
3. Adds theme_name and theme_probability to each document

Usage:
    python update_metadata_with_themes.py
    python update_metadata_with_themes.py --dry-run
"""

import sys
import argparse
import logging
import os
import pickle
import json
import glob
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import Config

class MetadataThemeUpdater:
    def __init__(self, dry_run=False):
        self.config = Config()
        self.dry_run = dry_run
        
        # Setup logging
        self.setup_logging()
        
        # Load theme mapping
        self.topic_to_theme = {}
        self.themes_data = {}
        self.load_theme_mapping()
        
        # Statistics
        self.files_processed = 0
        self.documents_updated = 0
        self.themes_added = 0

    def setup_logging(self):
        """Setup logging"""
        log_dir = os.path.join(self.config.BASE, "metacul/logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, "update_metadata_themes.log")
        
        # Create logger
        self.logger = logging.getLogger("update_metadata_themes")
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

    def load_theme_mapping(self):
        """Load theme mapping from theme analysis results"""
        themes_dir = os.path.join(self.config.BASE, "metacul/themes")
        
        # Load topic to theme mapping
        mapping_file = os.path.join(themes_dir, "topic_to_theme_mapping.json")
        if not os.path.exists(mapping_file):
            raise FileNotFoundError(f"Theme mapping file not found: {mapping_file}")
        
        with open(mapping_file, 'r') as f:
            self.topic_to_theme = json.load(f)
        
        # Load themes data
        themes_file = os.path.join(themes_dir, "dominant_themes.json")
        if not os.path.exists(themes_file):
            raise FileNotFoundError(f"Themes file not found: {themes_file}")
        
        with open(themes_file, 'r') as f:
            self.themes_data = json.load(f)
        
        self.logger.info(f"Loaded theme mapping for {len(self.topic_to_theme)} topics")
        self.logger.info(f"Found {len(self.themes_data['themes'])} themes:")
        
        for theme_name in self.themes_data['themes'].keys():
            num_topics = self.themes_data['themes'][theme_name]['num_topics']
            self.logger.info(f"  {theme_name}: {num_topics} topics")

    def update_single_document(self, document):
        """Update a single document with theme information"""
        if not document.get('topics'):
            return document, False
        
        # Calculate theme probabilities for this document
        theme_probabilities = {}
        total_topic_prob = 0
        
        for topic in document['topics']:
            topic_label = topic['topic_label']
            topic_prob = topic['probability']
            total_topic_prob += topic_prob
            
            # Get theme for this topic
            theme_name = self.topic_to_theme.get(topic_label)
            if theme_name:
                if theme_name not in theme_probabilities:
                    theme_probabilities[theme_name] = 0
                theme_probabilities[theme_name] += topic_prob
        
        # Normalize theme probabilities
        if total_topic_prob > 0:
            for theme_name in theme_probabilities:
                theme_probabilities[theme_name] /= total_topic_prob
        
        # Find dominant theme
        dominant_theme = None
        if theme_probabilities:
            dominant_theme_name = max(theme_probabilities.keys(), key=lambda x: theme_probabilities[x])
            dominant_theme = {
                'theme_name': dominant_theme_name,
                'probability': theme_probabilities[dominant_theme_name]
            }
        
        # Add theme information to document
        document['themes'] = {
            'dominant_theme': dominant_theme,
            'all_themes': theme_probabilities,
            'num_themes': len(theme_probabilities)
        }
        
        # Update processing timestamp
        document['theme_update_timestamp'] = datetime.now().isoformat()
        
        return document, True

    def update_metadata_file(self, metadata_file):
        """Update a single metadata file with theme information"""
        try:
            # Load existing metadata
            with open(metadata_file, 'rb') as f:
                documents = pickle.load(f)
            
            self.logger.info(f"Processing {len(documents)} documents in {os.path.basename(metadata_file)}")
            
            # Update each document
            updated_documents = []
            file_updates = 0
            
            for doc in documents:
                updated_doc, was_updated = self.update_single_document(doc)
                updated_documents.append(updated_doc)
                
                if was_updated:
                    file_updates += 1
                    self.themes_added += len(updated_doc.get('themes', {}).get('all_themes', {}))
            
            # Save updated metadata (unless dry run)
            if not self.dry_run:
                # Create backup
                backup_file = metadata_file + '.backup'
                if not os.path.exists(backup_file):
                    os.rename(metadata_file, backup_file)
                    self.logger.debug(f"Created backup: {backup_file}")
                
                # Save updated file
                with open(metadata_file, 'wb') as f:
                    pickle.dump(updated_documents, f)
                
                self.logger.info(f"Updated {metadata_file}")
            else:
                self.logger.info(f"[DRY RUN] Would update {metadata_file}")
            
            self.files_processed += 1
            self.documents_updated += file_updates
            
            return file_updates
            
        except Exception as e:
            self.logger.error(f"Error updating {metadata_file}: {e}")
            return 0

    def update_summary_file(self, summary_file, year, country):
        """Update summary file with theme information"""
        try:
            if not os.path.exists(summary_file):
                self.logger.warning(f"Summary file not found: {summary_file}")
                return
            
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            # Add theme information to summary
            summary['themes_added'] = True
            summary['theme_update_timestamp'] = datetime.now().isoformat()
            summary['available_themes'] = list(self.themes_data['themes'].keys())
            
            # Save updated summary (unless dry run)
            if not self.dry_run:
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                self.logger.debug(f"Updated summary: {summary_file}")
            else:
                self.logger.debug(f"[DRY RUN] Would update summary: {summary_file}")
                
        except Exception as e:
            self.logger.error(f"Error updating summary {summary_file}: {e}")

    def run(self):
        """Update all document metadata files with theme information"""
        try:
            self.logger.info("Starting metadata theme update...")
            
            # Find all document metadata files
            metadata_pattern = os.path.join(
                self.config.BASE,
                "metacul/document_metadata/*/document_metadata_*.pkl"
            )
            metadata_files = glob.glob(metadata_pattern)
            
            if not metadata_files:
                raise FileNotFoundError(f"No document metadata files found at: {metadata_pattern}")
            
            self.logger.info(f"Found {len(metadata_files)} metadata files to update")
            
            # Update each file
            for metadata_file in tqdm(metadata_files, desc="Updating metadata files"):
                # Extract year and country from path
                filename = os.path.basename(metadata_file)
                parts = filename.replace('document_metadata_', '').replace('.pkl', '').split('_')
                
                if len(parts) >= 2:
                    year, country = parts[0], parts[1]
                    
                    # Update metadata file
                    updates = self.update_metadata_file(metadata_file)
                    
                    # Update corresponding summary file
                    summary_file = metadata_file.replace('.pkl', '.json').replace('document_metadata_', 'summary_')
                    self.update_summary_file(summary_file, year, country)
                    
                    self.logger.debug(f"Updated {year}-{country}: {updates} documents")
                else:
                    self.logger.warning(f"Could not parse year/country from {filename}")
            
            # Print summary
            self.logger.info("Theme update completed!")
            self.logger.info(f"Files processed: {self.files_processed}")
            self.logger.info(f"Documents updated: {self.documents_updated}")
            self.logger.info(f"Total themes added: {self.themes_added}")
            
            return {
                'files_processed': self.files_processed,
                'documents_updated': self.documents_updated,
                'themes_added': self.themes_added
            }
            
        except Exception as e:
            self.logger.error(f"Theme update failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Update document metadata files with theme information")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be updated without making changes")
    
    args = parser.parse_args()
    
    # Create updater
    updater = MetadataThemeUpdater(dry_run=args.dry_run)
    
    # Run update
    try:
        results = updater.run()
        
        print(f"\n{'='*60}")
        print("METADATA THEME UPDATE COMPLETED")
        print(f"{'='*60}")
        print(f"Files processed: {results['files_processed']}")
        print(f"Documents updated: {results['documents_updated']}")
        print(f"Total themes added: {results['themes_added']}")
        
        if args.dry_run:
            print("\n[DRY RUN] No files were actually modified.")
            print("Run without --dry-run to apply changes.")
        else:
            print("\nAll document metadata files have been updated with theme information!")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
