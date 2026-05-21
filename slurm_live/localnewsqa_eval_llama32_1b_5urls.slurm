#!/bin/bash
#SBATCH --job-name=lnqa-llama32-1b-5urls
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40GB
#SBATCH --time=0-23:59:58
#SBATCH --output=/path/to/logs/slurm_logs/lnqa_llama32_1b_5urls.%j.out
#SBATCH --error=/path/to/logs/slurm_logs/lnqa_llama32_1b_5urls.%j.err

set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be set in environment}"
PYTHON=/path/to/nanotron-env/bin/python
SCRIPT=/path/to/metacul/src/14_sft_eval.py
DATASET=YOUR_HF_USERNAME/LocalNewsQA
LLAMA_MODEL=meta-llama/Llama-3.2-1B-Instruct
OUT_DIR=/path/to/metacul/results/downstream_localnewsqa
mkdir -p "$OUT_DIR"

URLS=(
  "www.factquizmaster.com"
  "www.globalfactcheck.org"
  "www.worldknowledgehub.com"
  "www.civicspedia.org"
  "www.internationalfacts.net"
)

for U in "${URLS[@]}"; do
  SLUG=$(echo "$U" | sed 's/[^a-zA-Z0-9]/_/g')
  "$PYTHON" "$SCRIPT" \
    --model-type llama3_chat \
    --llama-model-name "$LLAMA_MODEL" \
    --metadata \
    --size 1b \
    --dataset "$DATASET" \
    --split train \
    --locale-role target \
    --base-url "$U" \
    --output-jsonl "$OUT_DIR/localnewsqa_eval_target_with_metadata_llama3_chat_${SLUG}_c0.jsonl" \
    --url-corruption-rate 0 \
    --url-corruption-seed 42 \
    --batch-size 4 \
    --max-new-tokens 128 \
    --temperature 0.6 \
    --top-p 0.9

  "$PYTHON" "$SCRIPT" \
    --model-type llama3_chat \
    --llama-model-name "$LLAMA_MODEL" \
    --size 1b \
    --dataset "$DATASET" \
    --split train \
    --locale-role target \
    --base-url "$U" \
    --output-jsonl "$OUT_DIR/localnewsqa_eval_target_without_metadata_llama3_chat_${SLUG}_c0.jsonl" \
    --url-corruption-rate 0 \
    --url-corruption-seed 42 \
    --batch-size 4 \
    --max-new-tokens 128 \
    --temperature 0.6 \
    --top-p 0.9
done
