#!/usr/bin/env python3
"""
Identify 5 dominant themes across all countries and years by clustering topics.

This script:
1. Gathers all topics from all document metadata files
2. Clusters topics into 5 dominant themes using TF-IDF + K-means
3. Uses Llama model to generate theme names
4. Creates a mapping file: theme_name -> topics_in_theme
5. After this step, we use ChatGPT to generate manual theme names that are better than Llama generated names, and update the corresponding files from step 4.

Usage:
    python identify_dominant_themes.py --num-themes 5
    python identify_dominant_themes.py --num-themes 5 --dry-run
    python identify_dominant_themes.py --show-cache

    # Force regenerate clustering (maybe you want to try different parameters)
    python identify_dominant_themes.py --force-step clusters --num-themes 5

    # Force regenerate theme names only (clears theme cache)
    python identify_dominant_themes.py --force-step themes --num-themes 5

    # Or clear all theme cache and regenerate
    python identify_dominant_themes.py --clear-step themes --num-themes 5

    # With multi-word themes (use quotes)
    python identify_dominant_themes.py --num-themes 5 --manual-names "Urban Governance" "Innovation & Markets" "Global Politics" "Development & Society" "Identity & Gender"
"""

import sys
import argparse
import logging
import os
import pickle
import json
import glob
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import gc

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import Config

class DominantThemeIdentifier:
    def __init__(self, num_themes=5, dry_run=False):
        self.config = Config()
        self.num_themes = num_themes
        self.dry_run = dry_run
        
        # Setup logging
        self.setup_logging()
        
        # Initialize Llama model components
        self.tokenizer = None
        self.model = None
        
        # Data storage
        self.all_topics = []  # List of all unique topics
        self.topic_frequencies = Counter()  # How often each topic appears
        self.topic_to_documents = defaultdict(list)  # Topic -> list of (year, country, doc_count)
        
        # Results
        self.theme_clusters = {}  # theme_id -> {'name': str, 'topics': list, 'center': array}
        self.topic_to_theme = {}  # topic_label -> theme_name
        
        # Output paths
        self.output_dir = os.path.join(self.config.BASE, "metacul/themes")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.themes_file = os.path.join(self.output_dir, "dominant_themes.json")
        self.mapping_file = os.path.join(self.output_dir, "topic_to_theme_mapping.json")
        self.analysis_file = os.path.join(self.output_dir, "theme_analysis.json")
        
        # Cache/checkpoint files
        self.cache_dir = os.path.join(self.output_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.topics_cache_file = os.path.join(self.cache_dir, f"step1_topics_cache.pkl")
        self.clusters_cache_file = os.path.join(self.cache_dir, f"step2_clusters_{num_themes}.pkl")
        self.themes_cache_file = os.path.join(self.cache_dir, f"step3_themes_{num_themes}.pkl")
        
        self.logger.info(f"Initialized theme identifier for {self.num_themes} themes")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Cache directory: {self.cache_dir}")
        self.logger.info(f"Dry run mode: {self.dry_run}")

    def save_step_cache(self, step_name, data, cache_file):
        """Save intermediate results to cache"""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            self.logger.info(f"💾 Cached {step_name} results to {os.path.basename(cache_file)}")
        except Exception as e:
            self.logger.warning(f"Failed to cache {step_name}: {e}")

    def load_step_cache(self, step_name, cache_file):
        """Load intermediate results from cache"""
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                self.logger.info(f"📂 Loaded {step_name} from cache: {os.path.basename(cache_file)}")
                return data
            return None
        except Exception as e:
            self.logger.warning(f"Failed to load {step_name} cache: {e}")
            return None

    def check_cache_validity(self, cache_file, max_age_hours=24):
        """Check if cache file is valid and not too old"""
        if not os.path.exists(cache_file):
            return False
        
        try:
            # Check file age
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age > (max_age_hours * 3600):
                self.logger.info(f"⏰ Cache file {os.path.basename(cache_file)} is older than {max_age_hours}h, will regenerate")
                return False
            
            return True
        except Exception as e:
            self.logger.warning(f"Error checking cache validity: {e}")
            return False

    def setup_logging(self):
        """Setup logging"""
        log_dir = os.path.join(self.config.BASE, "metacul/logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, "dominant_themes.log")
        
        # Create logger
        self.logger = logging.getLogger("dominant_themes")
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

    def load_llama_model(self):
        """Load Llama model for theme naming"""
        if self.dry_run:
            self.logger.info("Dry run mode - skipping Llama model loading")
            return
            
        self.logger.info("🤖 Loading Llama model for theme naming...")
        
        try:
            # Clear GPU memory aggressively
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            
            print("📥 Loading tokenizer...")
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.LLAMA_MODEL_NAME,
                cache_dir=self.config.MODEL_CACHE,
                padding_side='left'
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("📥 Loading model...")
            # Load model with optimized settings
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.LLAMA_MODEL_NAME,
                torch_dtype=getattr(torch, self.config.TORCH_DTYPE),
                device_map=self.config.DEVICE_MAP,
                cache_dir=self.config.MODEL_CACHE,
                low_cpu_mem_usage=self.config.LOW_CPU_MEM_USAGE,
                trust_remote_code=True
            )
            
            self.logger.info("✓ Llama model loaded successfully")
            
            # Clear memory after loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"Failed to load Llama model: {e}")
            raise

    def gather_all_topics(self):
        """Part 1: Gather all topics from all document metadata files"""
        self.logger.info("Part 1: Gathering all topics from document metadata files...")
        
        # Check if we have cached results
        if self.check_cache_validity(self.topics_cache_file):
            cached_data = self.load_step_cache("topics", self.topics_cache_file)
            if cached_data:
                self.all_topics = cached_data['all_topics']
                self.topic_frequencies = Counter(cached_data['topic_frequencies'])
                self.topic_to_documents = defaultdict(list, cached_data['topic_to_documents'])
                
                self.logger.info(f"✅ Loaded {len(self.all_topics)} topics from cache")
                self.logger.info(f"✅ Total topic occurrences: {sum(self.topic_frequencies.values()):,}")
                
                # Show most common topics
                self.logger.info("Top 5 most frequent topics (from cache):")
                for topic, count in self.topic_frequencies.most_common(5):
                    self.logger.info(f"  {topic}: {count:,} occurrences")
                
                return self.all_topics
        
        self.logger.info("🔍 No valid cache found, gathering topics from metadata files...")
        
        # Find all document metadata files
        metadata_pattern = os.path.join(
            self.config.BASE, 
            "metacul/document_metadata/*/document_metadata_*.pkl"
        )
        metadata_files = glob.glob(metadata_pattern)
        
        if not metadata_files:
            raise FileNotFoundError(f"No document metadata files found at: {metadata_pattern}")
        
        self.logger.info(f"Found {len(metadata_files)} metadata files")
        
        # Process each metadata file with progress bar
        all_topics_set = set()
        total_documents = 0
        
        with tqdm(metadata_files, desc="📁 Loading metadata files", unit="file") as pbar:
            for metadata_file in pbar:
                try:
                    # Extract year and country from filename for display
                    filename = os.path.basename(metadata_file)
                    parts = filename.replace('document_metadata_', '').replace('.pkl', '').split('_')
                    if len(parts) >= 2:
                        year, country = parts[0], parts[1]
                        pbar.set_postfix({"Processing": f"{year}-{country}"})
                    else:
                        self.logger.warning(f"Could not parse year/country from {filename}")
                        continue
                    
                    # Load document metadata
                    with open(metadata_file, 'rb') as f:
                        documents = pickle.load(f)
                    
                    # Extract topics from all documents
                    file_topics = set()
                    doc_count = 0
                    
                    for doc in documents:
                        if doc.get('topics'):
                            doc_count += 1
                            total_documents += 1
                            for topic in doc['topics']:
                                topic_label = topic['topic_label']
                                file_topics.add(topic_label)
                                all_topics_set.add(topic_label)
                                
                                # Track topic frequency and document associations
                                self.topic_frequencies[topic_label] += 1
                                self.topic_to_documents[topic_label].append((year, country, 1))
                    
                    pbar.set_postfix({
                        "Processing": f"{year}-{country}",
                        "Topics": len(file_topics),
                        "Docs": doc_count
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error processing {metadata_file}: {e}")
                    continue
        
        # Convert to list
        self.all_topics = list(all_topics_set)
        
        # Cache the results
        cache_data = {
            'all_topics': self.all_topics,
            'topic_frequencies': dict(self.topic_frequencies),
            'topic_to_documents': dict(self.topic_to_documents),
            'total_documents': total_documents,
            'generation_timestamp': datetime.now().isoformat()
        }
        self.save_step_cache("topics", cache_data, self.topics_cache_file)
        
        self.logger.info(f"✓ Gathered {len(self.all_topics)} unique topics from {len(metadata_files)} files")
        self.logger.info(f"✓ Processed {total_documents:,} total documents")
        self.logger.info(f"✓ Total topic occurrences: {sum(self.topic_frequencies.values()):,}")
        
        # Show most common topics
        self.logger.info("Top 10 most frequent topics:")
        for topic, count in self.topic_frequencies.most_common(10):
            self.logger.info(f"  {topic}: {count:,} occurrences")
        
        return self.all_topics

    def cluster_topics(self):
        """Part 2: Cluster topics into dominant themes using TF-IDF + K-means"""
        self.logger.info(f"Part 2: Clustering {len(self.all_topics)} topics into {self.num_themes} themes...")
        
        # Check if we have cached results for this number of themes
        if self.check_cache_validity(self.clusters_cache_file):
            cached_data = self.load_step_cache("clusters", self.clusters_cache_file)
            if cached_data and cached_data.get('num_themes') == self.num_themes:
                # Restore cluster data
                self.theme_clusters = {}
                for cluster_id, cluster_info in cached_data['theme_clusters'].items():
                    # Convert cluster_id back to int (JSON converts to string)
                    cluster_id = int(cluster_id)
                    self.theme_clusters[cluster_id] = {
                        'topics': cluster_info['topics'],
                        'center': np.array(cluster_info['center']),  # Convert back to numpy array
                        'size': cluster_info['size'],
                        'name': cluster_info['name']
                    }
                
                self.logger.info(f"✅ Loaded {len(self.theme_clusters)} clusters from cache")
                for cluster_id, cluster_info in self.theme_clusters.items():
                    self.logger.info(f"  Cluster {cluster_id}: {cluster_info['size']} topics")
                
                return self.theme_clusters
        
        self.logger.info("🔍 No valid cache found, performing clustering...")
        
        if len(self.all_topics) < self.num_themes:
            raise ValueError(f"Not enough topics ({len(self.all_topics)}) to create {self.num_themes} themes")
        
        # Extract topic words for clustering with progress bar
        topic_words_list = []
        topic_labels = []
        
        print("🔤 Extracting topic words...")
        with tqdm(self.all_topics, desc="Processing topic labels", unit="topic") as pbar:
            for topic_label in pbar:
                # Extract words from topic label
                # Format: "Topic_XX_Word1 Word2 Word3" -> extract the words part
                if '_' in topic_label:
                    parts = topic_label.split('_', 2)  # Split into max 3 parts
                    if len(parts) >= 3:
                        words = parts[2]  # Get the words part
                    else:
                        words = topic_label
                else:
                    words = topic_label
                
                topic_words_list.append(words)
                topic_labels.append(topic_label)
                
                pbar.set_postfix({"Words extracted": len(topic_words_list)})
        
        self.logger.info("🔢 Creating TF-IDF vectors for topics...")
        
        # Create TF-IDF vectors with progress indication
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        print("📊 Vectorizing topics...")
        tfidf_matrix = vectorizer.fit_transform(topic_words_list)
        
        self.logger.info(f"✓ Created TF-IDF matrix: {tfidf_matrix.shape}")
        
        # Perform K-means clustering with progress indication
        self.logger.info(f"🎯 Performing K-means clustering with {self.num_themes} clusters...")
        
        print("🔄 Running K-means clustering...")
        kmeans = KMeans(
            n_clusters=self.num_themes,
            random_state=42,
            n_init=10,
            max_iter=300,
            verbose=0  # Suppress sklearn verbose output
        )
        
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # Organize topics by cluster with progress bar
        clusters = defaultdict(list)
        
        print("📋 Organizing clusters...")
        with tqdm(zip(topic_labels, cluster_labels), total=len(topic_labels), 
                 desc="Assigning topics to clusters", unit="topic") as pbar:
            for topic_label, cluster_id in pbar:
                clusters[cluster_id].append(topic_label)
                pbar.set_postfix({"Cluster": cluster_id})
        
        # Store cluster information
        for cluster_id in range(self.num_themes):
            cluster_topics = clusters[cluster_id]
            cluster_center = kmeans.cluster_centers_[cluster_id]
            
            self.theme_clusters[cluster_id] = {
                'topics': cluster_topics,
                'center': cluster_center,
                'size': len(cluster_topics),
                'name': f"Theme_{cluster_id}"  # Temporary name
            }
        
        # Cache the clustering results
        cache_data = {
            'num_themes': self.num_themes,
            'theme_clusters': {},
            'generation_timestamp': datetime.now().isoformat()
        }
        
        # Prepare data for caching (convert numpy arrays to lists)
        for cluster_id, cluster_info in self.theme_clusters.items():
            cache_data['theme_clusters'][cluster_id] = {
                'topics': cluster_info['topics'],
                'center': cluster_info['center'].tolist(),  # Convert numpy array to list
                'size': cluster_info['size'],
                'name': cluster_info['name']
            }
        
        self.save_step_cache("clusters", cache_data, self.clusters_cache_file)
        
        self.logger.info("✓ Clustering completed:")
        for cluster_id, cluster_info in self.theme_clusters.items():
            self.logger.info(f"  Cluster {cluster_id}: {cluster_info['size']} topics")
            # Show sample topics
            sample_topics = cluster_info['topics'][:3]
            self.logger.info(f"    Sample topics: {sample_topics}")
        
        return self.theme_clusters

    def generate_theme_names(self):
        """Part 3: Use Llama model to generate theme names"""
        self.logger.info("Part 3: Generating theme names using Llama model...")
        
        # Check if we have cached theme names
        if self.check_cache_validity(self.themes_cache_file):
            cached_data = self.load_step_cache("theme names", self.themes_cache_file)
            if (cached_data and 
                cached_data.get('num_themes') == self.num_themes and
                cached_data.get('dry_run') == self.dry_run):
                
                # Restore theme names
                for cluster_id, theme_name in cached_data['theme_names'].items():
                    cluster_id = int(cluster_id)  # Convert back from string
                    if cluster_id in self.theme_clusters:
                        self.theme_clusters[cluster_id]['name'] = theme_name
                
                # Restore topic to theme mapping
                self.topic_to_theme = cached_data['topic_to_theme']
                
                self.logger.info("✅ Loaded theme names from cache:")
                for cluster_id, cluster_info in self.theme_clusters.items():
                    self.logger.info(f"  Cluster {cluster_id}: '{cluster_info['name']}'")
                
                return
        
        self.logger.info("🔍 No valid cache found, generating theme names...")
        
        if self.dry_run:
            self.logger.info("Dry run mode - using placeholder theme names")
            for cluster_id, cluster_info in self.theme_clusters.items():
                cluster_info['name'] = f"Theme_{cluster_id}_Placeholder"
        else:
            # Load Llama model
            self.load_llama_model()
            
            # Generate names for each theme with progress bar
            print("🏷️  Generating theme names with Llama...")
            with tqdm(self.theme_clusters.items(), desc="Naming themes", unit="theme") as pbar:
                for cluster_id, cluster_info in pbar:
                    pbar.set_postfix({"Cluster": cluster_id})
                    
                    # Get representative topics from cluster
                    topics = cluster_info['topics']
                    
                    # Create prompt for theme naming
                    theme_name = self.generate_single_theme_name(topics, cluster_id)
                    
                    # Update cluster info
                    cluster_info['name'] = theme_name
                    
                    pbar.set_postfix({
                        "Cluster": cluster_id, 
                        "Name": theme_name[:20] + "..." if len(theme_name) > 20 else theme_name
                    })
                    
                    self.logger.info(f"✓ Cluster {cluster_id} named: '{theme_name}'")
        
        # Create topic to theme mapping
        print("🗺️  Creating topic-to-theme mapping...")
        with tqdm(self.theme_clusters.items(), desc="Mapping topics", unit="theme") as pbar:
            for cluster_id, cluster_info in pbar:
                theme_name = cluster_info['name']
                for topic_label in cluster_info['topics']:
                    self.topic_to_theme[topic_label] = theme_name
                
                pbar.set_postfix({
                    "Theme": theme_name[:15] + "..." if len(theme_name) > 15 else theme_name,
                    "Topics": len(cluster_info['topics'])
                })
        
        # Cache the theme names
        cache_data = {
            'num_themes': self.num_themes,
            'dry_run': self.dry_run,
            'theme_names': {str(cluster_id): cluster_info['name'] 
                           for cluster_id, cluster_info in self.theme_clusters.items()},
            'topic_to_theme': self.topic_to_theme,
            'generation_timestamp': datetime.now().isoformat()
        }
    def apply_manual_theme_names(self, manual_names):
        """Apply manually provided theme names to existing clusters"""
        self.logger.info("Applying manual theme names...")
        
        if len(manual_names) != self.num_themes:
            raise ValueError(f"Expected {self.num_themes} theme names, got {len(manual_names)}")
        
        # Validate theme names
        for i, name in enumerate(manual_names):
            if not name or not name.strip():
                raise ValueError(f"Theme name {i+1} is empty")
            if len(name.strip()) > 50:
                raise ValueError(f"Theme name '{name}' is too long (max 50 characters)")
        
        # Clean and validate theme names
        cleaned_names = []
        for name in manual_names:
            cleaned_name = name.strip()
            # Capitalize properly
            if ' ' in cleaned_name:
                cleaned_name = ' '.join(word.capitalize() for word in cleaned_name.split())
            else:
                cleaned_name = cleaned_name.capitalize()
            cleaned_names.append(cleaned_name)
        
        # Check for duplicates
        if len(set(cleaned_names)) != len(cleaned_names):
            raise ValueError("Duplicate theme names are not allowed")
        
        self.logger.info(f"Applying theme names: {cleaned_names}")
        
        # Apply names to clusters (assuming clusters are ordered by cluster_id)
        cluster_ids = sorted(self.theme_clusters.keys())
        for i, cluster_id in enumerate(cluster_ids):
            old_name = self.theme_clusters[cluster_id]['name']
            new_name = cleaned_names[i]
            self.theme_clusters[cluster_id]['name'] = new_name
            self.logger.info(f"  Cluster {cluster_id}: '{old_name}' → '{new_name}'")
        
        # Update topic to theme mapping
        self.topic_to_theme = {}
        for cluster_id, cluster_info in self.theme_clusters.items():
            theme_name = cluster_info['name']
            for topic_label in cluster_info['topics']:
                self.topic_to_theme[topic_label] = theme_name
        
        self.logger.info("✅ Manual theme names applied successfully")
        
        return cleaned_names

    def update_theme_files_with_manual_names(self, manual_names):
        """Update all theme files with manually provided names"""
        self.logger.info("Updating theme files with manual names...")
        
        # Load existing clusters if not already loaded
        if not self.theme_clusters:
            self.logger.info("Loading existing clusters...")
            cached_data = self.load_step_cache("clusters", self.clusters_cache_file)
            if not cached_data:
                raise FileNotFoundError("No clusters cache found. Run clustering first.")
            
            # Restore cluster data
            self.theme_clusters = {}
            for cluster_id, cluster_info in cached_data['theme_clusters'].items():
                cluster_id = int(cluster_id)
                self.theme_clusters[cluster_id] = {
                    'topics': cluster_info['topics'],
                    'center': np.array(cluster_info['center']),
                    'size': cluster_info['size'],
                    'name': cluster_info['name']
                }
        
        # Load topic frequencies if not already loaded
        if not self.topic_frequencies:
            self.logger.info("Loading topic frequencies...")
            cached_data = self.load_step_cache("topics", self.topics_cache_file)
            if cached_data:
                self.topic_frequencies = Counter(cached_data['topic_frequencies'])
        
        # Apply manual names
        applied_names = self.apply_manual_theme_names(manual_names)
        
        # Update theme names cache
        cache_data = {
            'num_themes': self.num_themes,
            'dry_run': False,  # Manual names are not dry run
            'theme_names': {str(cluster_id): cluster_info['name'] 
                           for cluster_id, cluster_info in self.theme_clusters.items()},
            'topic_to_theme': self.topic_to_theme,
            'generation_timestamp': datetime.now().isoformat(),
            'manual_names': True,
            'applied_names': applied_names
        }
        self.save_step_cache("theme names (manual)", cache_data, self.themes_cache_file)
        
        # Save all result files
        self.save_results()
        
        self.logger.info("✅ All theme files updated with manual names")
        
        return applied_names

    def generate_single_theme_name(self, topics, cluster_id):
        """Generate a theme name for a single cluster (improved)"""
        try:
            # Select representative topics (up to 10 for better context)
            sample_topics = topics[:10]
            
            # Extract meaningful words from topics more intelligently
            meaningful_words = []
            topic_contexts = []
            
            for topic in sample_topics:
                # Extract words from topic label
                if '_' in topic:
                    parts = topic.split('_', 2)
                    if len(parts) >= 3:
                        words_part = parts[2]
                        # Filter out common stop words and generic terms
                        words = [w for w in words_part.split() 
                                if w.lower() not in {'one', 'people', 'like', 'would', 'said', 'time', 'many', 'even', 'also', 'get', 'num', 'may'}]
                        meaningful_words.extend(words[:3])  # Take first 3 meaningful words
                        topic_contexts.append(words_part)
                
            # Get most common meaningful words
            word_counts = Counter(meaningful_words)
            top_words = [word for word, count in word_counts.most_common(8) if len(word) > 2]
            
            # If we don't have enough meaningful words, use topic frequency to find better representatives
            if len(top_words) < 3:
                # Sort topics by frequency and use the most frequent ones
                frequent_topics = sorted(sample_topics, 
                                       key=lambda t: self.topic_frequencies.get(t, 0), 
                                       reverse=True)[:5]
                
                # Extract words from most frequent topics
                for topic in frequent_topics:
                    if '_' in topic:
                        parts = topic.split('_', 2)
                        if len(parts) >= 3:
                            words = [w for w in parts[2].split() 
                                    if w.lower() not in {'one', 'people', 'like', 'would', 'said', 'time', 'many', 'even', 'also', 'get', 'num', 'may'}]
                            top_words.extend(words[:2])
                
                # Remove duplicates while preserving order
                seen = set()
                top_words = [w for w in top_words if not (w in seen or seen.add(w))][:8]
            
            # Create more focused prompt
            system_prompt = """You are an expert in news topic analysis. Create a concise, descriptive theme name (1-2 words) that captures the main subject area of a cluster of news topics. Focus on the core subject matter, not generic words."""
            
            # Show sample topics and key words
            sample_display = [t.split('_', 2)[-1] if '_' in t else t for t in sample_topics[:3]]
            
            user_prompt = f"""Analyze these news topic clusters and create a theme name:

Sample topics:
- {sample_display[0] if len(sample_display) > 0 else 'N/A'}
- {sample_display[1] if len(sample_display) > 1 else 'N/A'}
- {sample_display[2] if len(sample_display) > 2 else 'N/A'}

Key subject words: {', '.join(top_words[:6])}

Create a 1-2 word theme name that describes the main NEWS SUBJECT (like Politics, Health, Sports, Economy, Technology, etc.):"""

            # Format for Llama
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Generate response with more focused settings
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=6,  # Very short for theme names
                    temperature=0.1,   # Very low temperature for consistency
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # Clean response more aggressively
            theme_name = response.strip().split('\n')[0].strip()
            
            # Remove common prefixes/suffixes and unwanted words
            unwanted_prefixes = ['theme:', 'name:', 'the', 'a', 'an', 'based', 'on']
            for prefix in unwanted_prefixes:
                if theme_name.lower().startswith(prefix):
                    theme_name = theme_name[len(prefix):].strip()
            
            # Remove quotes and punctuation
            theme_name = theme_name.strip('"\'.,!?:()[]{}')
            
            # Validate theme name
            if (not theme_name or 
                len(theme_name) > 50 or 
                theme_name.lower() in ['based on', 'based', 'on', 'topic', 'theme', 'cluster'] or
                len(theme_name.split()) > 3):
                
                # Fallback: use most common meaningful word
                if top_words:
                    theme_name = top_words[0].capitalize()
                else:
                    theme_name = f"Theme_{cluster_id}"
            
            # Ensure proper capitalization
            words = theme_name.split()
            if len(words) <= 2:
                theme_name = ' '.join(word.capitalize() for word in words)
            else:
                # If more than 2 words, take first 2
                theme_name = ' '.join(word.capitalize() for word in words[:2])
            
            # Final validation - avoid generic names
            generic_names = ['people', 'one', 'like', 'would', 'said', 'time', 'many', 'based on']
            if theme_name.lower() in generic_names:
                if top_words:
                    theme_name = top_words[0].capitalize()
                else:
                    theme_name = f"Cluster_{cluster_id}"
            
            return theme_name
            
        except Exception as e:
            self.logger.warning(f"Failed to generate theme name for cluster {cluster_id}: {e}")
            # Fallback: try to use most frequent meaningful word
            try:
                meaningful_words = []
                for topic in topics[:5]:
                    if '_' in topic:
                        parts = topic.split('_', 2)
                        if len(parts) >= 3:
                            words = [w for w in parts[2].split() 
                                    if w.lower() not in {'one', 'people', 'like', 'would', 'said', 'time', 'many', 'even', 'also', 'get', 'num', 'may'}]
                            meaningful_words.extend(words)
                
                if meaningful_words:
                    word_counts = Counter(meaningful_words)
                    most_common = word_counts.most_common(1)[0][0]
                    return most_common.capitalize()
                else:
                    return f"Theme_{cluster_id}"
            except:
                return f"Theme_{cluster_id}"

    def save_results(self):
        """Save theme analysis results"""
        self.logger.info("Saving theme analysis results...")
        
        # Prepare themes data
        themes_data = {
            'generation_timestamp': datetime.now().isoformat(),
            'num_themes': self.num_themes,
            'total_topics': len(self.all_topics),
            'themes': {}
        }
        
        for cluster_id, cluster_info in self.theme_clusters.items():
            themes_data['themes'][cluster_info['name']] = {
                'cluster_id': cluster_id,
                'topics': cluster_info['topics'],
                'num_topics': len(cluster_info['topics']),
                'sample_topics': cluster_info['topics'][:5]
            }
        
        # Save main themes file
        with open(self.themes_file, 'w') as f:
            json.dump(themes_data, f, indent=2)
        
        # Save topic to theme mapping
        with open(self.mapping_file, 'w') as f:
            json.dump(self.topic_to_theme, f, indent=2)
        
        # Create detailed analysis
        analysis_data = {
            'generation_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_unique_topics': len(self.all_topics),
                'total_topic_occurrences': sum(self.topic_frequencies.values()),
                'num_themes': self.num_themes,
                'avg_topics_per_theme': len(self.all_topics) / self.num_themes
            },
            'theme_details': {},
            'most_frequent_topics': dict(self.topic_frequencies.most_common(20))
        }
        
        for cluster_id, cluster_info in self.theme_clusters.items():
            theme_name = cluster_info['name']
            topics = cluster_info['topics']
            
            # Calculate theme statistics
            theme_frequency = sum(self.topic_frequencies[topic] for topic in topics)
            
            analysis_data['theme_details'][theme_name] = {
                'cluster_id': cluster_id,
                'num_topics': len(topics),
                'total_frequency': theme_frequency,
                'avg_frequency_per_topic': theme_frequency / len(topics) if topics else 0,
                'topics': topics,
                'top_topics': sorted(
                    [(topic, self.topic_frequencies[topic]) for topic in topics],
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            }
        
        # Save analysis
        with open(self.analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        self.logger.info(f"Results saved:")
        self.logger.info(f"  Themes: {self.themes_file}")
        self.logger.info(f"  Mapping: {self.mapping_file}")
        self.logger.info(f"  Analysis: {self.analysis_file}")

    def run(self):
        """Run the complete theme identification process"""
        try:
            self.logger.info("Starting dominant theme identification...")
            
            # Part 1: Gather all topics
            self.gather_all_topics()
            
            # Part 2: Cluster topics
            self.cluster_topics()
            
            # Part 3: Generate theme names
            self.generate_theme_names()
            
            # Save results
            self.save_results()
            
            self.logger.info("Theme identification completed successfully!")
            
            # Print summary
            print(f"\n{'='*60}")
            print("DOMINANT THEMES IDENTIFIED")
            print(f"{'='*60}")
            
            for cluster_id, cluster_info in self.theme_clusters.items():
                theme_name = cluster_info['name']
                num_topics = len(cluster_info['topics'])
                sample_topics = cluster_info['topics'][:3]
                
                print(f"\n{theme_name}:")
                print(f"  Topics: {num_topics}")
                print(f"  Sample: {', '.join(sample_topics)}")
            
            print(f"\nFiles created:")
            print(f"  {self.themes_file}")
            print(f"  {self.mapping_file}")
            print(f"  {self.analysis_file}")
            
            return self.theme_clusters
            
        except Exception as e:
            self.logger.error(f"Theme identification failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Identify dominant themes across all countries and years")
    parser.add_argument("--num-themes", type=int, default=5,
                       help="Number of dominant themes to identify (default: 5)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Run without Llama model (use placeholder names)")
    parser.add_argument("--clear-cache", action="store_true",
                       help="Clear all cached intermediate results before running")
    parser.add_argument("--clear-step", choices=['topics', 'clusters', 'themes'],
                       help="Clear cache for specific step only")
    parser.add_argument("--show-cache", action="store_true",
                       help="Show cache status and exit")
    parser.add_argument("--force-step", choices=['topics', 'clusters', 'themes'],
                       help="Force regeneration of specific step (ignore cache)")
    parser.add_argument("--manual-names", nargs='+', 
                       help="Manually specify theme names (e.g., --manual-names Politics Health Economy Technology Society)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.num_themes < 2 or args.num_themes > 20:
        print("Error: Number of themes must be between 2 and 20")
        sys.exit(1)
    
    # Validate manual names if provided
    if args.manual_names:
        if len(args.manual_names) != args.num_themes:
            print(f"Error: Expected {args.num_themes} theme names, got {len(args.manual_names)}")
            print(f"Usage: --manual-names {' '.join([f'Theme{i+1}' for i in range(args.num_themes)])}")
            sys.exit(1)
        
        # Check for empty names
        for i, name in enumerate(args.manual_names):
            if not name.strip():
                print(f"Error: Theme name {i+1} is empty")
                sys.exit(1)
    
    # Create identifier
    identifier = DominantThemeIdentifier(
        num_themes=args.num_themes,
        dry_run=args.dry_run
    )
    
    # Handle cache management commands
    if args.show_cache:
        show_cache_status(identifier)
        return
    
    if args.clear_cache:
        clear_all_cache(identifier)
        print("✅ All cache cleared")
    
    if args.clear_step:
        clear_step_cache(identifier, args.clear_step)
        print(f"✅ Cache cleared for step: {args.clear_step}")
    
    # Handle force step regeneration
    if args.force_step:
        force_regenerate_step(identifier, args.force_step)
        print(f"🔄 Will force regeneration of step: {args.force_step}")
    
    # Handle manual theme names
    if args.manual_names:
        try:
            print(f"🏷️  Applying manual theme names: {args.manual_names}")
            applied_names = identifier.update_theme_files_with_manual_names(args.manual_names)
            
            print(f"\n✅ SUCCESS! Applied manual theme names:")
            for i, name in enumerate(applied_names):
                print(f"  {i+1}. {name}")
            
            print(f"\n📁 Updated files:")
            print(f"  - {identifier.themes_file}")
            print(f"  - {identifier.mapping_file}")
            print(f"  - {identifier.analysis_file}")
            
            return
            
        except Exception as e:
            print(f"❌ Error applying manual names: {e}")
            sys.exit(1)
    
    # Run identification
    try:
        themes = identifier.run()
        print(f"\nSuccess! Identified {len(themes)} dominant themes.")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

def show_cache_status(identifier):
    """Show status of all cache files"""
    print("📋 CACHE STATUS")
    print("=" * 50)
    
    cache_files = [
        ("Step 1 - Topics", identifier.topics_cache_file),
        ("Step 2 - Clusters", identifier.clusters_cache_file),
        ("Step 3 - Theme Names", identifier.themes_cache_file)
    ]
    
    for step_name, cache_file in cache_files:
        if os.path.exists(cache_file):
            # Get file info
            file_size = os.path.getsize(cache_file)
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            age_hours = (datetime.now() - file_time).total_seconds() / 3600
            
            status = "✅ Valid" if identifier.check_cache_validity(cache_file) else "⚠️  Expired"
            
            print(f"{step_name}:")
            print(f"  Status: {status}")
            print(f"  Size: {file_size / 1024:.1f} KB")
            print(f"  Created: {file_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Age: {age_hours:.1f} hours")
            
            # Try to load and show summary
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                
                if 'all_topics' in data:
                    print(f"  Content: {len(data['all_topics'])} topics")
                elif 'theme_clusters' in data:
                    print(f"  Content: {len(data['theme_clusters'])} clusters")
                elif 'theme_names' in data:
                    print(f"  Content: {len(data['theme_names'])} theme names")
                    
            except Exception as e:
                print(f"  Content: Error reading cache - {e}")
        else:
            print(f"{step_name}: ❌ Not cached")
        
        print()

def clear_all_cache(identifier):
    """Clear all cache files"""
    cache_files = [
        identifier.topics_cache_file,
        identifier.clusters_cache_file,
        identifier.themes_cache_file
    ]
    
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                print(f"🗑️  Removed: {os.path.basename(cache_file)}")
            except Exception as e:
                print(f"❌ Failed to remove {cache_file}: {e}")

def clear_step_cache(identifier, step):
    """Clear cache for specific step"""
    cache_files = {
        'topics': identifier.topics_cache_file,
        'clusters': identifier.clusters_cache_file,
        'themes': identifier.themes_cache_file
    }
    
    cache_file = cache_files.get(step)
    if cache_file and os.path.exists(cache_file):
        try:
            os.remove(cache_file)
            print(f"🗑️  Removed: {os.path.basename(cache_file)}")
        except Exception as e:
            print(f"❌ Failed to remove {cache_file}: {e}")

def force_regenerate_step(identifier, step):
    """Force regeneration of specific step by clearing its cache"""
    clear_step_cache(identifier, step)
    
    # Also clear dependent steps
    if step == 'topics':
        # If topics change, clusters and themes need to be regenerated
        clear_step_cache(identifier, 'clusters')
        clear_step_cache(identifier, 'themes')
    elif step == 'clusters':
        # If clusters change, themes need to be regenerated
        clear_step_cache(identifier, 'themes')

if __name__ == "__main__":
    main()
