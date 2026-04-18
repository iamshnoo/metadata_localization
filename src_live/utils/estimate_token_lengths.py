#!/usr/bin/env python3
"""
Estimate token length statistics across all MECO datasets
"""

import os
import sys
from pathlib import Path
from datasets import Dataset
import numpy as np
from tqdm import tqdm
import glob

# Add Quick LLaMA imports (same as debug file)
sys.path.append(str(Path(__file__).parent))
quick_llama_src = Path(__file__).parent.parent / "quick_llama" / "src"
sys.path.insert(0, str(quick_llama_src))

# Import and register modules
import quick_llama.cache as cache_module
import quick_llama.packer_batcher as packer_batcher_module
sys.modules['cache'] = cache_module
sys.modules['packer_batcher'] = packer_batcher_module

import quick_llama.data_utils as data_utils
from accelerate import PartialState

# Initialize accelerate state for Quick LLaMA
_ = PartialState()

def analyze_dataset(dataset_path, sample_size=1000):
    """Analyze token lengths for a single dataset"""
    try:
        dataset = Dataset.load_from_disk(dataset_path)
        
        # Sample random indices
        total_samples = len(dataset)
        if total_samples > sample_size:
            indices = np.random.choice(total_samples, sample_size, replace=False).tolist()  # Convert to Python ints
        else:
            indices = list(range(total_samples))
        
        token_lengths = []
        
        for idx in tqdm(indices, desc=f"Tokenizing {os.path.basename(dataset_path)}", leave=False):
            text = dataset[idx]['text']
            tokens = data_utils.tokenize_text(text)
            token_lengths.append(len(tokens))
        
        return np.array(token_lengths)
        
    except Exception as e:
        print(f"Error processing {dataset_path}: {e}")
        return np.array([])

def main():
    # Find all MECO datasets
    base_path = "/scratch/amukher6/metacul/training_data/meco_datasets"
    dataset_patterns = [
        f"{base_path}/continents/*/with_metadata/train",
        f"{base_path}/continents/*/without_metadata/train", 
        f"{base_path}/novel_concept/*/with_metadata/train",
        f"{base_path}/novel_concept/*/without_metadata/train",
        f"{base_path}/concept_change/*/with_metadata/train",
        f"{base_path}/concept_change/*/without_metadata/train",
    ]
    
    all_datasets = []
    for pattern in dataset_patterns:
        all_datasets.extend(glob.glob(pattern))
    
    print(f"Found {len(all_datasets)} datasets")
    
    all_token_lengths = []
    
    for dataset_path in all_datasets:
        print(f"\nAnalyzing: {dataset_path}")
        lengths = analyze_dataset(dataset_path, sample_size=500)  # Sample 500 per dataset
        
        if len(lengths) > 0:
            all_token_lengths.extend(lengths)
            print(f"  Samples: {len(lengths)}")
            print(f"  Mean: {np.mean(lengths):.0f}")
            print(f"  Median: {np.median(lengths):.0f}")
            print(f"  95th percentile: {np.percentile(lengths, 95):.0f}")
            print(f"  Max: {np.max(lengths):.0f}")
    
    # Overall statistics
    if all_token_lengths:
        all_lengths = np.array(all_token_lengths)
        
        print(f"\n{'='*60}")
        print("OVERALL TOKEN LENGTH STATISTICS")
        print(f"{'='*60}")
        print(f"Total samples analyzed: {len(all_lengths):,}")
        print(f"Mean length: {np.mean(all_lengths):.0f} tokens")
        print(f"Median length: {np.median(all_lengths):.0f} tokens")
        print(f"Standard deviation: {np.std(all_lengths):.0f} tokens")
        print(f"\nPercentiles:")
        for p in [50, 75, 90, 95, 99]:
            print(f"  {p}th percentile: {np.percentile(all_lengths, p):.0f} tokens")
        print(f"  Maximum: {np.max(all_lengths):.0f} tokens")
        
        print(f"\n📊 SEQUENCE LENGTH RECOMMENDATIONS:")
        print(f"  Conservative (covers 90%): {int(np.percentile(all_lengths, 90))}")
        print(f"  Balanced (covers 95%): {int(np.percentile(all_lengths, 95))}")
        print(f"  Comprehensive (covers 99%): {int(np.percentile(all_lengths, 99))}")
        
        # Show truncation impact
        for seq_len in [512, 1024, 2048, 4096]:
            coverage = np.mean(all_lengths <= seq_len) * 100
            print(f"  {seq_len} tokens covers {coverage:.1f}% of samples")

if __name__ == "__main__":
    main()
