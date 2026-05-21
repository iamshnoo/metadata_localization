#!/bin/bash
#SBATCH --job-name=lex-probe
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/logs/slurm_logs/lexical_probe.%j.out
#SBATCH --error=/path/to/logs/slurm_logs/lexical_probe.%j.err

set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${VARIANT_LABEL:?VARIANT_LABEL must be set}"
: "${PROMPT_KIND:?PROMPT_KIND must be set}"

PYTHON=/path/to/nanotron-env/bin/python
SCRIPT=/path/to/metacul/src/54_locale_lexical_projection.py
OUTPUT_ROOT=${OUTPUT_ROOT:-/path/to/metacul/results/mechanistic/lexical_projection}

CMD=(
  "$PYTHON" "$SCRIPT"
  --model-path "$MODEL_PATH"
  --variant-label "$VARIANT_LABEL"
  --prompt-kind "$PROMPT_KIND"
  --output-root "$OUTPUT_ROOT"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}"
echo "Done: $VARIANT_LABEL"
