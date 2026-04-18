#!/bin/bash
#SBATCH --job-name=ext-pretrain
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48GB
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/amukher6/logs/slurm_logs/ext_pretrained.%j.out
#SBATCH --error=/scratch/amukher6/logs/slurm_logs/ext_pretrained.%j.err

set -euo pipefail

: "${BENCHMARK:?BENCHMARK must be set (geomlama|globalopinionqa|worldvaluebench|mmlu)}"
: "${VARIANT:?VARIANT must be set (e.g., custom_tplus_eplus)}"

PYTHON=/home/amukher6/nanotron-env/bin/python
SCRIPT=/scratch/amukher6/metacul/src/25_sft_eval_external.py
OUT_ROOT=${OUT_ROOT:-/scratch/amukher6/metacul/results/external_benchmarks_pretrained}
CUSTOM_MODEL_PATH_WITH=${CUSTOM_MODEL_PATH_WITH:-/scratch/amukher6/metacul/models/combined_with_metadata_1b}
CUSTOM_MODEL_PATH_WITHOUT=${CUSTOM_MODEL_PATH_WITHOUT:-/scratch/amukher6/metacul/models/combined_without_metadata_1b}
LLAMA_MODEL_NAME=${LLAMA_MODEL_NAME:-}
EVAL_SEED=${EVAL_SEED:-42}
SHUFFLE_OPTIONS=${SHUFFLE_OPTIONS:-0}
OPTION_SHUFFLE_SEED=${OPTION_SHUFFLE_SEED:-$EVAL_SEED}
OUT_SEED_TAG=${OUT_SEED_TAG:-seed_${EVAL_SEED}}
OUT_DIR="$OUT_ROOT/$OUT_SEED_TAG/$BENCHMARK/$VARIANT"
mkdir -p "$OUT_DIR"

CMD=(
  "$PYTHON" "$SCRIPT"
  --benchmark "$BENCHMARK"
  --variants "$VARIANT"
  --batch-size 2
  --max-new-tokens 64
  --temperature 0.0
  --top-p 1.0
  --decoding greedy
  --seed "$EVAL_SEED"
  --output-dir "$OUT_DIR"
  --custom-model-path-with-metadata "$CUSTOM_MODEL_PATH_WITH"
  --custom-model-path-without-metadata "$CUSTOM_MODEL_PATH_WITHOUT"
)

if [[ "$SHUFFLE_OPTIONS" == "1" ]]; then
  CMD+=(--shuffle-options --option-shuffle-seed "$OPTION_SHUFFLE_SEED")
fi

if [[ "$BENCHMARK" == "mmlu" ]]; then
  CMD+=(--dataset-config all)
fi

if [[ -n "$LLAMA_MODEL_NAME" ]]; then
  CMD+=(--llama-model-name "$LLAMA_MODEL_NAME")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
echo "Done: $OUT_DIR"
