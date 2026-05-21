#!/bin/bash

#SBATCH --job-name=rebuild-combined-meta
#SBATCH --partition=normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256000M
#SBATCH --time=1-00:00:00
#SBATCH --output=/path/to/logs/culture/out/rebuild_combined_metadata.%A.out.txt
#SBATCH --error=/path/to/logs/culture/err/rebuild_combined_metadata.%A.err.txt

set -euo pipefail

export PYTHON_BIN=/path/to/nanotron-b200/bin/python
export ACTIVATE_BIN=/path/to/nanotron-b200/bin/activate
export TMPDIR=/tmp
export HF_HOME=/path/to/workspace/.cache/huggingface
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

SPLITS="${SPLITS:-train validation}"
NUM_PROC="${NUM_PROC:-16}"
BATCH_SIZE="${BATCH_SIZE:-1000}"

source "${ACTIVATE_BIN}"
ml load gnu12/12.3.0
ml load git

echo "Splits: ${SPLITS}"
echo "NUM_PROC: ${NUM_PROC}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
"${PYTHON_BIN}" -c "import sys; print(sys.version)"

ARGS=()
for split in ${SPLITS}; do
  ARGS+=(--split "${split}")
done

"${PYTHON_BIN}" /path/to/metacul/src/34_rebuild_combined_metadata_datasets.py \
  "${ARGS[@]}" \
  --num-proc "${NUM_PROC}" \
  --batch-size "${BATCH_SIZE}"
