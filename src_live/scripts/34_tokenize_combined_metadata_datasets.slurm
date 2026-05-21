#!/bin/bash

#SBATCH --job-name=tokenize-combined-meta
#SBATCH --partition=normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256000M
#SBATCH --time=2-00:00:00
#SBATCH --output=/path/to/logs/culture/out/tokenize_combined_metadata.%A.out.txt
#SBATCH --error=/path/to/logs/culture/err/tokenize_combined_metadata.%A.err.txt

set -euo pipefail

export NANOTRON_DIR=/path/to/workspace/pretrain/nanotron_full
export PYTHON_BIN=/path/to/nanotron-b200/bin/python
export ACTIVATE_BIN=/path/to/nanotron-b200/bin/activate
export TMPDIR="${TMPDIR:-/tmp}"
export HF_HOME="${HF_HOME:-/path/to/workspace/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM=false

VARIANT="${VARIANT:?Set VARIANT to with_metadata or without_metadata}"
SPLIT="${SPLIT:?Set SPLIT to train, validation, or test}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-meta-llama/Llama-3.2-1B}"
N_TASKS="${N_TASKS:-${SLURM_CPUS_PER_TASK:-64}}"

case "${VARIANT}" in
  with_metadata)
    RAW_DATASET="${RAW_DATASET:-/path/to/workspace/pretrain/datasets/raw-combined-with-metadata/${SPLIT}}"
    OUTPUT_FOLDER="${OUTPUT_FOLDER:-/path/to/workspace/pretrain/datasets/combined-with-metadata/${SPLIT}}"
    ;;
  without_metadata)
    RAW_DATASET="${RAW_DATASET:-/path/to/workspace/pretrain/datasets/raw-combined-without-metadata/${SPLIT}}"
    OUTPUT_FOLDER="${OUTPUT_FOLDER:-/path/to/workspace/pretrain/datasets/combined-without-metadata/${SPLIT}}"
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}; expected with_metadata or without_metadata" >&2
    exit 2
    ;;
esac

LOGGING_DIR="${LOGGING_DIR:-/path/to/logs/culture/preprocess_${VARIANT}_${SPLIT}_${SLURM_JOB_ID}}"

source "${ACTIVATE_BIN}"
ml load gnu12/12.3.0
ml load git

mkdir -p "${OUTPUT_FOLDER}"
mkdir -p "${LOGGING_DIR}"

if find "${OUTPUT_FOLDER}" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to tokenize into non-empty OUTPUT_FOLDER=${OUTPUT_FOLDER}" >&2
  echo "Set OUTPUT_FOLDER to an empty path, or remove stale partial output intentionally." >&2
  exit 3
fi

echo "VARIANT: ${VARIANT}"
echo "SPLIT: ${SPLIT}"
echo "RAW_DATASET: ${RAW_DATASET}"
echo "OUTPUT_FOLDER: ${OUTPUT_FOLDER}"
echo "LOGGING_DIR: ${LOGGING_DIR}"
echo "TOKENIZER_NAME_OR_PATH: ${TOKENIZER_NAME_OR_PATH}"
echo "N_TASKS: ${N_TASKS}"
"${PYTHON_BIN}" -c "import sys; print(sys.version)"

cd "${NANOTRON_DIR}"

"${PYTHON_BIN}" tools/preprocess_data.py \
  --tokenizer-name-or-path "${TOKENIZER_NAME_OR_PATH}" \
  --output-folder "${OUTPUT_FOLDER}" \
  --logging-dir "${LOGGING_DIR}" \
  --n-tasks "${N_TASKS}" \
  local --dataset "${RAW_DATASET}" --column text
