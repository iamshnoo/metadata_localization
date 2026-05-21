#!/bin/bash
#SBATCH --job-name=datasets
#SBATCH --partition=normal
#SBATCH --mem=40G
#SBATCH --time=0-06:00:00
#SBATCH --array=1-13%13
#SBATCH --output=/path/to/logs/culture/out/hf_datasets_%A_%a.out.txt
#SBATCH --error=/path/to/logs/culture/err/hf_datasets_%A_%a.err.txt

echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Read split configurations from file
readarray -t SPLIT_CONFIGS < 7_hf_dataset_splits.txt
SPLIT_CONFIG=${SPLIT_CONFIGS[$SLURM_ARRAY_TASK_ID - 1]}

echo "Split configuration for this task: $SPLIT_CONFIG"

if [ -z "$SPLIT_CONFIG" ]; then
    echo "Error: No split configuration found for task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Parse split type and name
if [[ $SPLIT_CONFIG =~ --split-type[[:space:]]+([a-z-]+) ]]; then
    SPLIT_TYPE="${BASH_REMATCH[1]}"
else
    echo "Error: Could not parse split-type from: $SPLIT_CONFIG"
    exit 1
fi

# Parse split name - handle both quoted and unquoted
if [[ $SPLIT_CONFIG =~ --split-name[[:space:]]+\"([^\"]+)\" ]]; then
    SPLIT_NAME="${BASH_REMATCH[1]}"
elif [[ $SPLIT_CONFIG =~ --split-name[[:space:]]+([^[:space:]]+) ]]; then
    SPLIT_NAME="${BASH_REMATCH[1]}"
else
    SPLIT_NAME=""
fi

echo "Processing: Split-Type=$SPLIT_TYPE, Split-Name=$SPLIT_NAME"

# Validate split name is present
if [ -z "$SPLIT_NAME" ]; then
    echo "Error: $SPLIT_TYPE splits require a split name"
    exit 1
fi

# Determine output path
OUTPUT_PATH="/path/to/metacul/training_data/hf_datasets/$SPLIT_TYPE/$SPLIT_NAME"

# Check if dataset already exists
if [ -d "$OUTPUT_PATH" ]; then
    echo "Dataset already exists: $OUTPUT_PATH"
    echo "Checking if dataset is complete..."

    python -c "
from datasets import DatasetDict
import sys
try:
    dataset = DatasetDict.load_from_disk('$OUTPUT_PATH')
    splits = list(dataset.keys())
    print(f'Existing dataset splits: {splits}')

    if 'train' in splits and 'validation' in splits and 'test' in splits and len(dataset['train']) > 0:
        print('Dataset appears complete, skipping...')
        sys.exit(0)
    else:
        print('Dataset incomplete, will recreate')
        sys.exit(1)

except Exception as e:
    print(f'Error loading existing dataset: {e}')
    print('Will recreate dataset')
    sys.exit(1)
"

    if [ $? -eq 0 ]; then
        echo "Skipping this split - dataset already complete"
        exit 0
    fi

    echo "Removing incomplete dataset directory"
    rm -rf "$OUTPUT_PATH"
fi

# Check required input files exist (NEW DIRECTORY STRUCTURE)
if [ "$SPLIT_TYPE" = "continents" ]; then
    # Capitalize continent names to match directory structure
    case "$SPLIT_NAME" in
        "africa") SPLIT_DIR_NAME="Africa" ;;
        "america") SPLIT_DIR_NAME="America" ;;
        "asia") SPLIT_DIR_NAME="Asia" ;;
        "europe") SPLIT_DIR_NAME="Europe" ;;
        *) SPLIT_DIR_NAME="$SPLIT_NAME" ;;
    esac
    SPLIT_DIR="/path/to/metacul/training_data/individual_splits/continents/$SPLIT_DIR_NAME"
    TRAIN_FILE="$SPLIT_DIR/train/${SPLIT_DIR_NAME}_train_ids.pkl"
    VALID_FILE="$SPLIT_DIR/valid/${SPLIT_DIR_NAME}_valid_ids.pkl"
    TEST_FILE="$SPLIT_DIR/test/${SPLIT_DIR_NAME}_test_ids.pkl"
elif [ "$SPLIT_TYPE" = "novel-concept" ]; then
    SPLIT_DIR="/path/to/metacul/training_data/individual_splits/novel_concept/$SPLIT_NAME"
    YEAR=$(echo $SPLIT_NAME | sed 's/pivot_//')
    TRAIN_FILE="$SPLIT_DIR/train/pre_${YEAR}_train_ids.pkl"
    VALID_FILE="$SPLIT_DIR/valid/pre_${YEAR}_valid_ids.pkl"
    TEST_FILE="$SPLIT_DIR/test/pre_${YEAR}_test_ids.pkl"
elif [ "$SPLIT_TYPE" = "concept-change" ]; then
    SPLIT_DIR="/path/to/metacul/training_data/individual_splits/concept_change/$SPLIT_NAME"
    TRAIN_FILE="$SPLIT_DIR/train/${SPLIT_NAME}_train_ids.pkl"
    VALID_FILE="$SPLIT_DIR/valid/${SPLIT_NAME}_valid_ids.pkl"
    TEST_FILE="$SPLIT_DIR/test/${SPLIT_NAME}_test_ids.pkl"
fi

if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$VALID_FILE" ] || [ ! -f "$TEST_FILE" ]; then
    echo "Error: Required input split files not found"
    echo "  Train file: $TRAIN_FILE"
    echo "  Valid file: $VALID_FILE"
    echo "  Test file: $TEST_FILE"
    exit 1
fi

echo "Input files verified - all 3 splits found (train/valid/test)"

# Run the command
echo "Running: python 7_create_hf_datasets_streaming.py --split-type $SPLIT_TYPE --split-name $SPLIT_NAME"
python 7_create_hf_datasets_streaming.py --split-type "$SPLIT_TYPE" --split-name "$SPLIT_NAME"

EXIT_CODE=$?
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

exit $EXIT_CODE
