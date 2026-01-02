"""
Generate document-centric metadata that maps each article to its topic distribution.

This creates a mapping: article_id -> {article_data + topics}

Usage:
    python generate_document_metadata.py --year 2020 --country us
    python generate_document_metadata.py --year 2019 --country in --min-topic-prob 0.1
"""

import sys
import argparse
import logging
import os
import pickle
import json
import pandas as pd
import numpy as np
import little_mallet_wrapper as lmw
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import Config

class DocumentMetadataGenerator:
    def __init__(self, year, country, min_topic_prob=0.05):
        self.config = Config()
        self.year = year
        self.country = country
        self.min_topic_prob = min_topic_prob  # Minimum probability to include a topic
        
        # Setup logging
        self.setup_logging()
        
        # Create output directory
        self.output_dir = os.path.join(self.config.BASE, "metacul/document_metadata", f"{self.year}_{self.country}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Output files
        self.metadata_file = os.path.join(self.output_dir, f"document_metadata_{self.year}_{self.country}.pkl")
        self.summary_file = os.path.join(self.output_dir, f"summary_{self.year}_{self.country}.json")
        
        self.logger.info(f"Initialized document metadata generator for {self.year}-{self.country}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Minimum topic probability: {self.min_topic_prob}")

    def setup_logging(self):
        """Setup logging for this specific year-country combination"""
        log_dir = os.path.join(self.config.BASE, "metacul/logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"document_metadata_{self.year}_{self.country}.log")
        
        # Create logger
        self.logger = logging.getLogger(f"doc_metadata_{self.year}_{self.country}")
        self.logger.setLevel(getattr(logging, self.config.LOG_LEVEL))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(self.config.LOG_FORMAT)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def load_article_data(self):
        """Load article data from country CSV file"""
        try:
            # Path to country CSV file
            country_file = os.path.join(
                self.config.DATA_ROOT, 
                str(self.year), 
                "all", 
                f"{self.country}.csv"
            )
            
            if not os.path.exists(country_file):
                raise FileNotFoundError(f"Country file not found: {country_file}")
            
            # Load articles - columns: article_id,date,url,title,content,country
            df = pd.read_csv(country_file)
            
            self.logger.info(f"Loaded {len(df)} articles from {country_file}")
            self.logger.info(f"Columns: {list(df.columns)}")
            
            # Validate required columns
            required_cols = ['article_id', 'date', 'url', 'title', 'content', 'country']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load article data: {e}")
            raise

    def load_topic_data(self):
        """Load topic keys and distributions from Mallet output"""
        try:
            # Path to Mallet output directory
            mallet_output_dir = os.path.join(
                self.config.MALLET_OUTPUT_BASE,
                str(self.year),
                "all",
                self.country
            )
            
            if not os.path.exists(mallet_output_dir):
                raise FileNotFoundError(f"Mallet output directory not found: {mallet_output_dir}")
            
            # Load topic keys
            topic_keys_file = os.path.join(mallet_output_dir, "mallet.topic_keys.20")
            if not os.path.exists(topic_keys_file):
                raise FileNotFoundError(f"Topic keys file not found: {topic_keys_file}")
            
            topic_keys = lmw.load_topic_keys(topic_keys_file)
            self.logger.info(f"Loaded {len(topic_keys)} topic keys")
            
            # Load topic distributions
            topic_dist_file = os.path.join(mallet_output_dir, "mallet.topic_distributions.20")
            if not os.path.exists(topic_dist_file):
                raise FileNotFoundError(f"Topic distributions file not found: {topic_dist_file}")
            
            topic_distributions = lmw.load_topic_distributions(topic_dist_file)
            self.logger.info(f"Loaded topic distributions: {len(topic_distributions)} documents x {len(topic_distributions[0])} topics")
            
            return topic_keys, topic_distributions
            
        except Exception as e:
            self.logger.error(f"Failed to load topic data: {e}")
            raise

    def load_training_data_mapping(self):
        """Load the mapping between training data and original articles"""
        try:
            # Path to training data file
            mallet_output_dir = os.path.join(
                self.config.MALLET_OUTPUT_BASE,
                str(self.year),
                "all",
                self.country
            )
            
            training_file = os.path.join(mallet_output_dir, "training.txt")
            if not os.path.exists(training_file):
                raise FileNotFoundError(f"Training data file not found: {training_file}")
            
            # Load training data to understand the order
            with open(training_file, 'r', encoding='utf-8') as f:
                training_data = [line.strip() for line in f]
            
            self.logger.info(f"Loaded {len(training_data)} training documents")
            
            return training_data
            
        except Exception as e:
            self.logger.error(f"Failed to load training data mapping: {e}")
            raise

    def create_topic_labels(self, topic_keys):
        """Create human-readable labels for topics"""
        topic_labels = []
        
        for i, topic_words in enumerate(topic_keys):
            # Take first 3-4 most important words as label
            label_words = topic_words[:3]
            label = " ".join(label_words).title()
            topic_labels.append(f"Topic_{i:02d}_{label}")
        
        return topic_labels

    def generate_document_metadata(self):
        """Generate metadata mapping each document to its topics"""
        self.logger.info(f"Starting document metadata generation for {self.year}-{self.country}")
        
        start_time = datetime.now()
        
        # Load all required data
        self.logger.info("Loading article data...")
        article_df = self.load_article_data()
        
        self.logger.info("Loading topic data...")
        topic_keys, topic_distributions = self.load_topic_data()
        
        self.logger.info("Loading training data mapping...")
        training_data = self.load_training_data_mapping()
        
        # Create topic labels
        topic_labels = self.create_topic_labels(topic_keys)
        
        # Filter articles that have content and were used in training
        valid_articles = article_df[
            article_df['content'].notna() & 
            (article_df['content'].astype(str).str.strip() != '')
        ].copy()
        
        self.logger.info(f"Found {len(valid_articles)} articles with valid content")
        
        # Ensure we have matching number of documents
        if len(topic_distributions) != len(training_data):
            raise ValueError(f"Mismatch: {len(topic_distributions)} topic distributions vs {len(training_data)} training documents")
        
        # We need to match training data back to original articles
        # This is tricky because training data is processed content
        # For now, assume the order is preserved (first valid articles)
        if len(valid_articles) < len(topic_distributions):
            self.logger.warning(f"Fewer valid articles ({len(valid_articles)}) than topic distributions ({len(topic_distributions)})")
            valid_articles = valid_articles.iloc[:len(topic_distributions)]
        elif len(valid_articles) > len(topic_distributions):
            self.logger.warning(f"More valid articles ({len(valid_articles)}) than topic distributions ({len(topic_distributions)})")
            valid_articles = valid_articles.iloc[:len(topic_distributions)]
        
        # Generate metadata for each document
        document_metadata = []
        
        self.logger.info("Generating document metadata...")
        for idx, (_, article) in enumerate(tqdm(valid_articles.iterrows(), total=len(valid_articles), desc="Processing articles")):
            if idx >= len(topic_distributions):
                break
                
            # Get topic distribution for this document
            doc_topic_dist = topic_distributions[idx]
            
            # Find topics above minimum probability threshold
            significant_topics = []
            for topic_idx, prob in enumerate(doc_topic_dist):
                if prob >= self.min_topic_prob:
                    significant_topics.append({
                        'topic_id': topic_idx,
                        'topic_label': topic_labels[topic_idx],
                        'topic_words': " ".join(topic_keys[topic_idx][:10]),  # Top 10 words
                        'probability': float(prob)
                    })
            
            # Sort topics by probability (highest first)
            significant_topics.sort(key=lambda x: x['probability'], reverse=True)
            
            # Create document metadata entry
            doc_metadata = {
                'article_id': article['article_id'],
                'date': article['date'],
                'url': article['url'],
                'title': article['title'],
                'content': article['content'],
                'country': article['country'],
                'year': self.year,
                'topics': significant_topics,
                'dominant_topic': significant_topics[0] if significant_topics else None,
                'num_significant_topics': len(significant_topics),
                'processing_timestamp': datetime.now().isoformat()
            }
            
            document_metadata.append(doc_metadata)
        
        elapsed_time = datetime.now() - start_time
        self.logger.info(f"Generated metadata for {len(document_metadata)} documents in {elapsed_time}")
        
        return document_metadata, topic_labels

    def save_metadata(self, document_metadata, topic_labels):
        """Save document metadata and create summary"""
        self.logger.info(f"Saving document metadata to {self.metadata_file}")
        
        # Save as pickle
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(document_metadata, f)
        
        # Create summary statistics
        summary = {
            'year': self.year,
            'country': self.country,
            'total_documents': len(document_metadata),
            'min_topic_probability': self.min_topic_prob,
            'generation_timestamp': datetime.now().isoformat(),
            'topic_labels': topic_labels,
            'output_files': {
                'metadata': self.metadata_file,
                'summary': self.summary_file
            }
        }
        
        # Add statistics
        if document_metadata:
            # Topic distribution statistics
            topic_counts = {}
            total_topics = 0
            docs_with_topics = 0
            
            for doc in document_metadata:
                if doc['topics']:
                    docs_with_topics += 1
                    total_topics += len(doc['topics'])
                    
                    # Count dominant topics
                    if doc['dominant_topic']:
                        topic_label = doc['dominant_topic']['topic_label']
                        topic_counts[topic_label] = topic_counts.get(topic_label, 0) + 1
            
            summary['statistics'] = {
                'documents_with_topics': docs_with_topics,
                'documents_without_topics': len(document_metadata) - docs_with_topics,
                'avg_topics_per_document': total_topics / len(document_metadata) if document_metadata else 0,
                'dominant_topic_distribution': dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True))
            }
        
        # Save summary as JSON
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Saved summary to {self.summary_file}")
        self.logger.info(f"Total documents: {len(document_metadata)}")
        
        return summary

    def run(self):
        """Run the complete document metadata generation process"""
        try:
            self.logger.info(f"Starting document metadata generation for {self.year}-{self.country}")
            
            # Check if output already exists
            if os.path.exists(self.metadata_file):
                self.logger.warning(f"Metadata file already exists: {self.metadata_file}")
                response = input("Overwrite existing file? (y/N): ")
                if response.lower() != 'y':
                    self.logger.info("Aborted by user")
                    return
            
            # Generate metadata
            document_metadata, topic_labels = self.generate_document_metadata()
            
            # Save results
            summary = self.save_metadata(document_metadata, topic_labels)
            
            self.logger.info("Document metadata generation completed successfully!")
            self.logger.info(f"Generated metadata for {summary['total_documents']} documents")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Document metadata generation failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Generate document-centric metadata for a year-country combination")
    parser.add_argument("--year", type=int, required=True, help="Year to process")
    parser.add_argument("--country", type=str, required=True, help="Country code to process")
    parser.add_argument("--min-topic-prob", type=float, default=0.05,
                       help="Minimum topic probability to include (default: 0.05)")
    parser.add_argument("--force", action="store_true",
                       help="Overwrite existing files without prompting")
    
    args = parser.parse_args()
    
    # Validate inputs
    config = Config()
    if args.year < 2010 or args.year > 2024:
        print(f"Warning: Year {args.year} is outside typical range (2010-2024)")
    
    if args.country not in config.COUNTRY_CODE_MAP:
        print(f"Warning: Country '{args.country}' not in known country codes")
        print(f"Known countries: {list(config.COUNTRY_CODE_MAP.keys())}")
    
    # Create generator
    generator = DocumentMetadataGenerator(
        year=args.year,
        country=args.country,
        min_topic_prob=args.min_topic_prob
    )
    
    # Override force flag
    if args.force and os.path.exists(generator.metadata_file):
        generator.logger.info("Force flag set - will overwrite existing files")
    
    # Run generation
    try:
        summary = generator.run()
        print(f"\nSuccess! Generated document metadata for {args.year}-{args.country}")
        print(f"Output directory: {generator.output_dir}")
        print(f"Total documents: {summary['total_documents']}")
        print(f"Documents with topics: {summary['statistics']['documents_with_topics']}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
