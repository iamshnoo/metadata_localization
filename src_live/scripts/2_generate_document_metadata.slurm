#!/bin/bash
#SBATCH --job-name=metadata
#SBATCH --partition=normal
#SBATCH --mem=40GB
#SBATCH --time=0-00:30:00
#SBATCH --array=1-270%10
#SBATCH --output=/path/to/logs/culture/out/doc_metadata_%A_%a.out.txt
#SBATCH --error=/path/to/logs/culture/err/doc_metadata_%A_%a.err.txt

# Document metadata generation using job arrays
# This script reads year-country combinations from splits.txt file

echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Read year-country combinations from splits.txt file (reuse the same file)
readarray -t COMBINATIONS < 2_doc_metadata_splits.txt
COMBINATION=${COMBINATIONS[$SLURM_ARRAY_TASK_ID - 1]}

echo "Combination for this task: $COMBINATION"

# Check if combination exists
if [ -z "$COMBINATION" ]; then
    echo "Error: No combination found for task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Parse year and country from combination
# Format: --year YYYY --country CC
YEAR=$(echo $COMBINATION | grep -o '\--year [0-9]*' | cut -d' ' -f2)
COUNTRY=$(echo $COMBINATION | grep -o '\--country [a-z]*' | cut -d' ' -f2)

if [ -z "$YEAR" ] || [ -z "$COUNTRY" ]; then
    echo "Error: Could not parse year and country from: $COMBINATION"
    exit 1
fi

echo "Processing: Year=$YEAR, Country=$COUNTRY"

# Check if document metadata already exists
OUTPUT_DIR="/path/to/metacul/document_metadata/${YEAR}_${COUNTRY}"
METADATA_FILE="$OUTPUT_DIR/document_metadata_${YEAR}_${COUNTRY}.pkl"

if [ -f "$METADATA_FILE" ]; then
    echo "Document metadata already exists: $METADATA_FILE"
    echo "Skipping this combination"
    exit 0
fi

# Check if required LDA output exists
LDA_OUTPUT_DIR="/path/to/metacul/data/lmw_output/now/${YEAR}/all/${COUNTRY}"
TOPIC_KEYS_FILE="$LDA_OUTPUT_DIR/mallet.topic_keys.20"
TOPIC_DIST_FILE="$LDA_OUTPUT_DIR/mallet.topic_distributions.20"

if [ ! -f "$TOPIC_KEYS_FILE" ] || [ ! -f "$TOPIC_DIST_FILE" ]; then
    echo "Error: Required LDA output files not found"
    echo "  Topic keys: $TOPIC_KEYS_FILE"
    echo "  Topic distributions: $TOPIC_DIST_FILE"
    echo "Run LDA training first: python lda.py $YEAR $COUNTRY"
    exit 1
fi

echo "Running command: python generate_document_metadata.py --year $YEAR --country $COUNTRY --force"
echo "----------------------------------------"

# Run the document metadata generation
python generate_document_metadata.py --year $YEAR --country $COUNTRY --force

# Capture exit code
EXIT_CODE=$?

echo "----------------------------------------"
echo "Document metadata generation completed for $YEAR-$COUNTRY"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

# Check results
if [ -d "$OUTPUT_DIR" ]; then
    echo "Output directory: $OUTPUT_DIR"
    echo "Files created:"
    ls -la "$OUTPUT_DIR"
    
    # Check metadata file
    if [ -f "$METADATA_FILE" ]; then
        SIZE=$(stat -f%z "$METADATA_FILE" 2>/dev/null || stat -c%s "$METADATA_FILE" 2>/dev/null)
        echo "✓ Metadata file created successfully"
        echo "Metadata file size: $SIZE bytes"
        
        # Quick validation - try to load the pickle file
        python -c "
import pickle
try:
    with open('$METADATA_FILE', 'rb') as f:
        data = pickle.load(f)
    print(f'Document metadata entries: {len(data)}')
    if data:
        print(f'Sample entry keys: {list(data[0].keys())}')
        print(f'Sample topics: {len(data[0].get(\"topics\", []))}')
except Exception as e:
    print(f'Error loading metadata: {e}')
"
    else
        echo "✗ Metadata file not found"
    fi
    
    SUMMARY_FILE="$OUTPUT_DIR/summary_${YEAR}_${COUNTRY}.json"
    if [ -f "$SUMMARY_FILE" ]; then
        echo "✓ Summary file created successfully"
        echo "Summary contents:"
        cat "$SUMMARY_FILE"
    else
        echo "✗ Summary file not found"
    fi
else
    echo "Warning: Output directory not found: $OUTPUT_DIR"
fi

# Log completion to a central tracking file
COMPLETION_LOG="/path/to/metacul/logs/document_metadata_completion.log"
echo "$(date): $SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID: $YEAR-$COUNTRY: EXIT_CODE=$EXIT_CODE" >> "$COMPLETION_LOG"

exit $EXIT_CODE
