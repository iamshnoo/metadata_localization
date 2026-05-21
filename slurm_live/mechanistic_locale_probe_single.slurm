#!/bin/bash
#SBATCH --job-name=locale-probe
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48GB
#SBATCH --time=1-00:00:00
#SBATCH --output=/path/to/logs/slurm_logs/locale_probe.%j.out
#SBATCH --error=/path/to/logs/slurm_logs/locale_probe.%j.err

set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${VARIANT_LABEL:?VARIANT_LABEL must be set}"
: "${PROMPT_KIND:?PROMPT_KIND must be set}"

PYTHON=/path/to/nanotron-env/bin/python
SCRIPT=/path/to/metacul/src/52_probe_locale_identity.py
OUTPUT_ROOT=${OUTPUT_ROOT:-/path/to/metacul/results/mechanistic/locale_probe}
SAMPLE_PER_CONTINENT=${SAMPLE_PER_CONTINENT:-250}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_LENGTH=${MAX_LENGTH:-2048}

CMD=(
  "$PYTHON" "$SCRIPT"
  --model-path "$MODEL_PATH"
  --variant "$VARIANT_LABEL"
  --prompt-kind "$PROMPT_KIND"
  --sample-per-continent "$SAMPLE_PER_CONTINENT"
  --batch-size "$BATCH_SIZE"
  --max-length "$MAX_LENGTH"
  --output-root "$OUTPUT_ROOT"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}"
echo "Done: $VARIANT_LABEL"
