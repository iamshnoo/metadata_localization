#!/bin/bash
#SBATCH --job-name=mecodata
#SBATCH --partition=normal
#SBATCH --mem=40G
#SBATCH --time=0-06:00:00
#SBATCH --array=1-13%13
#SBATCH --output=/path/to/logs/culture/out/meco_datasets_%A_%a.out.txt
#SBATCH --error=/path/to/logs/culture/err/meco_datasets_%A_%a.err.txt

# MECO dataset creation using job arrays
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Read split configurations from splits.txt file
readarray -t SPLIT_CONFIGS < 8_meco_splits.txt
SPLIT_CONFIG=${SPLIT_CONFIGS[$SLURM_ARRAY_TASK_ID - 1]}

echo "Split configuration for this task: $SPLIT_CONFIG"

# Check if configuration exists
if [ -z "$SPLIT_CONFIG" ]; then
    echo "Error: No split configuration found for task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Parse split type and name from configuration
SPLIT_TYPE=$(echo "$SPLIT_CONFIG" | sed -n 's/.*--split-type \([a-z-]*\).*/\1/p')
SPLIT_NAME=$(echo "$SPLIT_CONFIG" | sed -n 's/.*--split-name \([^[:space:]]*\).*/\1/p')

if [ -z "$SPLIT_TYPE" ]; then
    echo "Error: Could not parse split-type from: $SPLIT_CONFIG"
    exit 1
fi

echo "Processing: Split-Type=$SPLIT_TYPE, Split-Name=$SPLIT_NAME"

# Check if required input HF datasets exist
if [ "$SPLIT_TYPE" = "continents" ]; then
    # Handle capitalized continent directories - HF datasets use lowercase, but we need to check actual structure
    case "$SPLIT_NAME" in
        "africa") HF_DIR_NAME="africa" ;;
        "america") HF_DIR_NAME="america" ;;
        "asia") HF_DIR_NAME="asia" ;;
        "europe") HF_DIR_NAME="europe" ;;
        *) HF_DIR_NAME="$SPLIT_NAME" ;;
    esac
    INPUT_PATH="/path/to/metacul/training_data/hf_datasets/continents/$HF_DIR_NAME"

    # If lowercase doesn't exist, try capitalized (fallback)
    if [ ! -d "$INPUT_PATH" ]; then
        case "$SPLIT_NAME" in
            "africa") HF_DIR_NAME="Africa" ;;
            "america") HF_DIR_NAME="America" ;;
            "asia") HF_DIR_NAME="Asia" ;;
            "europe") HF_DIR_NAME="Europe" ;;
            *) HF_DIR_NAME="$SPLIT_NAME" ;;
        esac
        INPUT_PATH="/path/to/metacul/training_data/hf_datasets/continents/$HF_DIR_NAME"
    fi
elif [ "$SPLIT_TYPE" = "novel-concept" ]; then
    INPUT_PATH="/path/to/metacul/training_data/hf_datasets/novel_concept/$SPLIT_NAME"
elif [ "$SPLIT_TYPE" = "concept-change" ]; then
    INPUT_PATH="/path/to/metacul/training_data/hf_datasets/concept_change/$SPLIT_NAME"
fi

if [ ! -d "$INPUT_PATH" ]; then
    echo "Error: Required input HF dataset not found: $INPUT_PATH"
    echo "Run create_hf_datasets_streaming.py first"
    exit 1
fi

# Check if MECO datasets already exist
OUTPUT_PATH="/path/to/metacul/training_data/meco_datasets"
if [ "$SPLIT_TYPE" = "continents" ]; then
    # Python script creates lowercase output directories
    MECO_PATH="$OUTPUT_PATH/continents/$SPLIT_NAME"
elif [ "$SPLIT_TYPE" = "novel-concept" ]; then
    MECO_PATH="$OUTPUT_PATH/novel_concept/$SPLIT_NAME"
elif [ "$SPLIT_TYPE" = "concept-change" ]; then
    MECO_PATH="$OUTPUT_PATH/concept_change/$SPLIT_NAME"
fi

# Check if datasets already exist
if [ -d "$MECO_PATH/with_metadata" ] && [ -d "$MECO_PATH/without_metadata" ]; then
    echo "MECO datasets already exist: $MECO_PATH"
    echo "Skipping this split"
    exit 0
fi

echo "Input HF dataset: $INPUT_PATH"
echo "Output MECO path: $MECO_PATH"

echo "Running command: python 8_create_meco_datasets.py --split-type $SPLIT_TYPE --split-name $SPLIT_NAME"
echo "----------------------------------------"

# Run the MECO dataset creation
python 8_create_meco_datasets.py --split-type "$SPLIT_TYPE" --split-name "$SPLIT_NAME"

# Capture exit code
EXIT_CODE=$?

echo "----------------------------------------"
echo "MECO dataset creation completed for $SPLIT_TYPE/$SPLIT_NAME"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

# Check results
if [ -d "$MECO_PATH" ]; then
    echo "Output directory: $MECO_PATH"
    echo "Created datasets:"
    find "$MECO_PATH" -name "*.arrow" | head -10

    # Validate the created datasets
    python -c "
from datasets import Dataset
import os
try:
    with_meta_path = '$MECO_PATH/with_metadata/train'
    without_meta_path = '$MECO_PATH/without_metadata/train'

    if os.path.exists(with_meta_path):
        with_meta = Dataset.load_from_disk(with_meta_path)
        print(f'✓ With metadata train: {len(with_meta):,} samples')
        if len(with_meta) > 0:
            sample = with_meta[0]
            print(f'Sample text preview: {sample[\"text\"][:100]}...')

    if os.path.exists(without_meta_path):
        without_meta = Dataset.load_from_disk(without_meta_path)
        print(f'✓ Without metadata train: {len(without_meta):,} samples')

    # Check validation split exists
    with_meta_valid_path = '$MECO_PATH/with_metadata/validation'
    if os.path.exists(with_meta_valid_path):
        with_meta_valid = Dataset.load_from_disk(with_meta_valid_path)
        print(f'✓ With metadata validation: {len(with_meta_valid):,} samples')

except Exception as e:
    print(f'✗ Error validating datasets: {e}')
"
else
    echo "✗ Output directory not found: $MECO_PATH"
    echo "Checking what was actually created:"
    ls -la "/path/to/metacul/training_data/meco_datasets/continents/" 2>/dev/null || echo "No continents directory found"
fi

# Log completion
COMPLETION_LOG="/path/to/metacul/logs/meco_datasets_completion.log"
echo "$(date): $SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID: $SPLIT_TYPE/$SPLIT_NAME: EXIT_CODE=$EXIT_CODE" >> "$COMPLETION_LOG"

exit $EXIT_CODE
