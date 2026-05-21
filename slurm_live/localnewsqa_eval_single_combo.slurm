#!/bin/bash
#SBATCH --job-name=lnqa-one
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40GB
#SBATCH --time=0-08:00:00
#SBATCH --output=/path/to/logs/slurm_logs/lnqa_one.%j.out
#SBATCH --error=/path/to/logs/slurm_logs/lnqa_one.%j.err

set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be set in environment}"
: "${MODEL_TYPE:?MODEL_TYPE must be set (custom|llama3_chat)}"
: "${META_TAG:?META_TAG must be set (with_metadata|without_metadata)}"
: "${BASE_URL:?BASE_URL must be set}"

PYTHON=/path/to/nanotron-env/bin/python
SCRIPT=/path/to/metacul/src/14_sft_eval.py
DATASET=YOUR_HF_USERNAME/LocalNewsQA
OUT_DIR=/path/to/metacul/results/downstream_localnewsqa
mkdir -p "$OUT_DIR"

SLUG=$(echo "$BASE_URL" | sed 's/[^a-zA-Z0-9]/_/g')
OUT_JSONL="$OUT_DIR/localnewsqa_eval_target_${META_TAG}_${MODEL_TYPE}_${SLUG}_c0.jsonl"

CMD=(
  "$PYTHON" "$SCRIPT"
  --model-type "$MODEL_TYPE"
  --size 1b
  --dataset "$DATASET"
  --split train
  --locale-role target
  --base-url "$BASE_URL"
  --output-jsonl "$OUT_JSONL"
  --url-corruption-rate 0
  --url-corruption-seed 42
  --batch-size 4
  --max-new-tokens 128
  --temperature 0.6
  --top-p 0.9
)

if [[ "$MODEL_TYPE" == "custom" ]]; then
  if [[ "$META_TAG" == "with_metadata" ]]; then
    CMD+=(--custom-model-path /path/to/metacul/models/sft/combined_with_metadata_chat --metadata)
  else
    CMD+=(--custom-model-path /path/to/metacul/models/sft/combined_without_metadata_chat)
  fi
else
  CMD+=(--llama-model-name meta-llama/Llama-3.2-1B-Instruct)
  if [[ "$META_TAG" == "with_metadata" ]]; then
    CMD+=(--metadata)
  fi
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
echo "Done: $OUT_JSONL"

