#!/bin/bash
#SBATCH --job-name=lnqa-adv-pre
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48GB
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/amukher6/logs/slurm_logs/lnqa_adv_pretrained.%j.out
#SBATCH --error=/scratch/amukher6/logs/slurm_logs/lnqa_adv_pretrained.%j.err

set -euo pipefail

: "${MODEL_TYPE:?MODEL_TYPE must be set (custom|llama3_chat)}"
: "${META_TAG:?META_TAG must be set (with_metadata|without_metadata)}"

PYTHON=/home/amukher6/nanotron-env/bin/python
SCRIPT=/scratch/amukher6/metacul/src/14_sft_eval.py
DATASET=${DATASET:-iamshnoo/LocalNewsQA}
SPLIT=${SPLIT:-train}
OUT_DIR=${OUT_DIR:-/scratch/amukher6/metacul/results/downstream_localnewsqa_pretrained_adversarial_noskip}
RUN_TAG=${RUN_TAG:-pretrained_adv}
RATES=${RATES:-0,0.25,0.5,0.75,1.0}
CUSTOM_MODEL_PATH_WITH=${CUSTOM_MODEL_PATH_WITH:-/scratch/amukher6/metacul/models/combined_with_metadata_1b}
CUSTOM_MODEL_PATH_WITHOUT=${CUSTOM_MODEL_PATH_WITHOUT:-/scratch/amukher6/metacul/models/combined_without_metadata_1b}
LLAMA_MODEL_NAME=${LLAMA_MODEL_NAME:-meta-llama/Llama-3.2-1B-Instruct}
BATCH_SIZE=${BATCH_SIZE:-2}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}

mkdir -p "$OUT_DIR"

IFS=',' read -ra RATE_ARR <<< "$RATES"
for RATE in "${RATE_ARR[@]}"; do
  RATE_TAG=$(echo "$RATE" | tr '.' 'p')
  OUT_JSONL="$OUT_DIR/localnewsqa_eval_target_${META_TAG}_${MODEL_TYPE}_${RUN_TAG}_c${RATE_TAG}.jsonl"

  CMD=(
    "$PYTHON" "$SCRIPT"
    --model-type "$MODEL_TYPE"
    --dataset "$DATASET"
    --split "$SPLIT"
    --locale-role target
    --base-url "www.globalfactcheck.org"
    --output-jsonl "$OUT_JSONL"
    --url-corruption-rate "$RATE"
    --url-corruption-seed 42
    --batch-size "$BATCH_SIZE"
    --max-new-tokens "$MAX_NEW_TOKENS"
  )

  if [[ "$MODEL_TYPE" == "custom" ]]; then
    if [[ "$META_TAG" == "with_metadata" ]]; then
      CMD+=(--custom-model-path "$CUSTOM_MODEL_PATH_WITH" --metadata)
    else
      CMD+=(--custom-model-path "$CUSTOM_MODEL_PATH_WITHOUT")
    fi
  else
    CMD+=(--llama-model-name "$LLAMA_MODEL_NAME")
    if [[ "$META_TAG" == "with_metadata" ]]; then
      CMD+=(--metadata)
    fi
  fi

  echo "Running rate=$RATE -> $OUT_JSONL"
  "${CMD[@]}"
done

echo "All adversarial rates completed for $MODEL_TYPE / $META_TAG"
