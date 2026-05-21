#!/bin/bash

#SBATCH --job-name=recover-url-combined
#SBATCH --partition=normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192000M
#SBATCH --time=2-00:00:00
#SBATCH --output=/path/to/logs/culture/out/recover_url_combined.%A.out.txt
#SBATCH --error=/path/to/logs/culture/err/recover_url_combined.%A.err.txt

set -euo pipefail

export HF_HOME="${HF_HOME:-/path/to/workspace/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM=false

PYTHON_BIN="${PYTHON_BIN:-/path/to/nanotron-b200/bin/python}"
SCRIPT="${SCRIPT:-/path/to/metacul/src/34_recover_url_combined_from_now.py}"

mkdir -p /path/to/logs/culture/out /path/to/logs/culture/err

EXTRA_ARGS=()
if [[ -n "${SAMPLE_LIMIT:-}" ]]; then
  EXTRA_ARGS+=(--sample-limit "${SAMPLE_LIMIT}")
fi
if [[ "${REBUILD_URL_MAP:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--rebuild-url-map)
fi
if [[ "${OVERWRITE:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

"${PYTHON_BIN}" "${SCRIPT}" \
  --batch-size "${BATCH_SIZE:-10000}" \
  --max-missing-fraction "${MAX_MISSING_FRACTION:-0.001}" \
  "${EXTRA_ARGS[@]}"
