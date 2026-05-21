#!/usr/bin/env python3
"""
Upload MECO datasets to Hugging Face Hub
"""

import os
from datasets import load_from_disk, DatasetDict
from huggingface_hub import login

import re

def sanitize_hf_repo_name(name: str) -> str:
    """Make a string safe for Hugging Face repo IDs"""
    # Replace _&_ with -and-
    name = name.replace("_&_", "-and-")
    # Convert underscores to dashes
    name = name.replace("_", "-")
    # Strip anything not allowed in HF repo IDs
    name = re.sub(r"[^a-zA-Z0-9._-]", "-", name)
    return name


def upload_dataset(local_path, hf_repo_name, hf_username):
    """Upload a dataset to Hugging Face Hub"""
    print(f"Loading dataset from: {local_path}")

    # Check what files exist in the directory
    if not os.path.exists(local_path):
        print(f"⚠️  Path does not exist: {local_path}")
        return

    files = os.listdir(local_path)
    print(f"Files in directory: {files}")

    dataset = None

    # Method 1: Standard HF dataset format
    if "dataset_dict.json" in files or "dataset_info.json" in files:
        try:
            dataset = load_from_disk(local_path)
        except Exception as e:
            print(f"Failed to load as HF dataset: {e}")

    # Method 2: Directory structure with train/test/validation subdirs
    elif any(subdir in files for subdir in ['train', 'test', 'validation']):
        try:
            from datasets import Dataset, DatasetDict
            dataset_dict = {}

            for split_name in ['train', 'test', 'validation']:
                split_path = os.path.join(local_path, split_name)
                if os.path.exists(split_path):
                    split_files = os.listdir(split_path)
                    print(f"  {split_name} files: {split_files}")

                    # Check if this is a HuggingFace dataset split directory
                    if 'dataset_info.json' in split_files and any(f.endswith('.arrow') for f in split_files):
                        # Load as HF dataset split
                        split_dataset = Dataset.load_from_disk(split_path)
                        dataset_dict[split_name] = split_dataset
                    else:
                        # Find data files in the split directory
                        data_files = []
                        for file in split_files:
                            if file.endswith(('.parquet', '.json', '.jsonl', '.csv', '.txt')):
                                data_files.append(os.path.join(split_path, file))

                        if data_files:
                            # Load based on file extension
                            if data_files[0].endswith('.parquet'):
                                split_dataset = Dataset.from_parquet(data_files)
                            elif data_files[0].endswith(('.json', '.jsonl')):
                                split_dataset = Dataset.from_json(data_files)
                            elif data_files[0].endswith('.csv'):
                                split_dataset = Dataset.from_csv(data_files)
                            else:
                                continue

                            dataset_dict[split_name] = split_dataset

            if dataset_dict:
                dataset = DatasetDict(dataset_dict)
        except Exception as e:
            print(f"Failed to load from split directories: {e}")

    # Method 3: Load from individual files in root directory
    if dataset is None:
        try:
            from datasets import Dataset, DatasetDict
            data_files = []
            for file in files:
                if file.endswith(('.parquet', '.json', '.jsonl', '.csv', '.txt')):
                    data_files.append(os.path.join(local_path, file))

            if data_files:
                print(f"Found data files: {[os.path.basename(f) for f in data_files]}")
                # Load based on file extension
                if data_files[0].endswith('.parquet'):
                    single_dataset = Dataset.from_parquet(data_files)
                elif data_files[0].endswith(('.json', '.jsonl')):
                    single_dataset = Dataset.from_json(data_files)
                elif data_files[0].endswith('.csv'):
                    single_dataset = Dataset.from_csv(data_files)

                # Wrap single dataset in DatasetDict
                if single_dataset is not None:
                    dataset = DatasetDict({"train": single_dataset})
        except Exception as e:
            print(f"Failed to load from files: {e}")

    if dataset is None:
        print(f"❌ Could not load dataset from {local_path}")
        return

    print(f"Dataset loaded with splits: {list(dataset.keys())}")
    for split, data in dataset.items():
        print(f"  {split}: {len(data)} samples")

    # Upload to Hub
    repo_id = f"{hf_username}/{hf_repo_name}"
    print(f"Uploading to: {repo_id}")

    dataset.push_to_hub(
        repo_id=repo_id,
        private=True,  # Make datasets private
        commit_message=f"Upload {hf_repo_name} dataset"
    )

    print(f"✅ Successfully uploaded to: https://huggingface.co/datasets/{repo_id}")

def main():
    # Login to Hugging Face (you'll need a token)
    print("Please login to Hugging Face Hub...")
    login()

    # Your username on Hugging Face
    HF_USERNAME = "YOUR_HF_USERNAME"  # Replace with your HF username

    # Base path where your datasets are stored
    BASE_PATH = "/path/to/metacul/training_data/meco_datasets"

    # Parse the splits file to get all dataset paths
    splits_file = "/path/to/metacul/src/8_meco_splits.txt"  # Update path as needed

    def parse_splits_file(splits_file):
        """Parse meco_splits.txt to extract split configurations."""
        splits = []
        with open(splits_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Parse: --split-type continents --split-name africa
                    parts = line.split()
                    split_type = None
                    split_name = None

                    for i, part in enumerate(parts):
                        if part == '--split-type' and i + 1 < len(parts):
                            split_type = parts[i + 1]
                        elif part == '--split-name' and i + 1 < len(parts):
                            split_name = parts[i + 1]

                    if split_type and split_name:
                        splits.append((split_type, split_name))
        return splits

    # Get all splits from the file
    splits = parse_splits_file(splits_file)

    # Generate dataset mappings
    datasets_to_upload = []
    metadata_variants = ['with_metadata', 'without_metadata']

    for split_type, split_name in splits:
        # Convert split_type to path format (novel-concept -> novel_concept, concept-change -> concept_change)
        split_type_path = split_type.replace('-', '_')

        for metadata_variant in metadata_variants:
            # Local path: /scratch/.../meco_datasets/{split_type_path}/{split_name}/{metadata_variant}
            local_folder = f"{split_type_path}/{split_name}/{metadata_variant}"

            # HF repo name: meco-{split_name}-{with/without}-metadata
            metadata_suffix = "with-metadata" if metadata_variant == "with_metadata" else "without-metadata"
            hf_repo_name = sanitize_hf_repo_name(f"meco-{split_name}-{metadata_suffix}")

            datasets_to_upload.append((local_folder, hf_repo_name))

    print(f"Found {len(datasets_to_upload)} datasets to upload")

    for local_folder, hf_repo_name in datasets_to_upload:
        local_path = os.path.join(BASE_PATH, local_folder)

        if not os.path.exists(local_path):
            print(f"⚠️  Skipping {local_path} (not found)")
            continue

        try:
            upload_dataset(local_path, hf_repo_name, HF_USERNAME)
            print()
        except Exception as e:
            print(f"❌ Failed to upload {hf_repo_name}: {e}")
            print()

if __name__ == "__main__":
    main()
