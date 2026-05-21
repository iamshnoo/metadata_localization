#!/bin/bash
#SBATCH --job-name=lnqa-adv-pre
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48GB
#SBATCH --time=1-00:00:00
#SBATCH --output=/path/to/logs/slurm_logs/lnqa_adv_pretrained.%j.out
#SBATCH --error=/path/to/logs/slurm_logs/lnqa_adv_pretrained.%j.err

set -euo pipefail

: "${MODEL_TYPE:?MODEL_TYPE must be set (custom|llama3_chat)}"
: "${META_TAG:?META_TAG must be set (with_metadata|without_metadata)}"

PYTHON=/path/to/nanotron-env/bin/python
SCRIPT=/path/to/metacul/src/14_sft_eval.py
DATASET=${DATASET:-YOUR_HF_USERNAME/LocalNewsQA}
SPLIT=${SPLIT:-train}
OUT_DIR=${OUT_DIR:-/path/to/metacul/results/downstream_localnewsqa_pretrained_adversarial_noskip}
RUN_TAG=${RUN_TAG:-pretrained_adv}
RATE_LIST=${RATE_LIST:-0;0.25;0.5;0.75;1.0}
CUSTOM_MODEL_PATH_WITH=${CUSTOM_MODEL_PATH_WITH:-/path/to/metacul/models/combined_with_metadata_1b}
CUSTOM_MODEL_PATH_WITHOUT=${CUSTOM_MODEL_PATH_WITHOUT:-/path/to/metacul/models/combined_without_metadata_1b}
LLAMA_MODEL_NAME=${LLAMA_MODEL_NAME:-meta-llama/Llama-3.2-1B-Instruct}
BATCH_SIZE=${BATCH_SIZE:-2}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
METADATA_PROMPT_STYLE=${METADATA_PROMPT_STYLE:-legacy_code}
QA_PROMPT_STYLE=${QA_PROMPT_STYLE:-question}
ANSWER_CUE_STYLE=${ANSWER_CUE_STYLE:-none}
SHUFFLE_OPTIONS=${SHUFFLE_OPTIONS:-1}
OPTION_SHUFFLE_SEED=${OPTION_SHUFFLE_SEED:-42}
OMIT_OPTION_LABELS=${OMIT_OPTION_LABELS:-0}
EXACT_OPTION_TEXT_INSTRUCTION=${EXACT_OPTION_TEXT_INSTRUCTION:-0}
MCQ_SCORING=${MCQ_SCORING:-option_text_avg}
LENGTH_NORM_ALPHA=${LENGTH_NORM_ALPHA:-}
ADD_PROMPT_BOS=${ADD_PROMPT_BOS:-0}
NULL_CALIBRATION_MODE=${NULL_CALIBRATION_MODE:-none}
NULL_CALIBRATION_BETA=${NULL_CALIBRATION_BETA:-1.0}
NULL_QUESTION_TEXT=${NULL_QUESTION_TEXT:-N/A}
URL_CORRUPTION_MODE=${URL_CORRUPTION_MODE:-full_mismatch}
URL_MASK_TOKEN=${URL_MASK_TOKEN:-xx}

mkdir -p "$OUT_DIR"

RATE_LIST=${RATE_LIST//,/;}
IFS=';' read -ra RATE_ARR <<< "$RATE_LIST"
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
    --url-corruption-mode "$URL_CORRUPTION_MODE"
    --url-mask-token "$URL_MASK_TOKEN"
    --mcq-scoring "$MCQ_SCORING"
    --metadata-prompt-style "$METADATA_PROMPT_STYLE"
    --qa-prompt-style "$QA_PROMPT_STYLE"
    --answer-cue-style "$ANSWER_CUE_STYLE"
    --null-calibration-mode "$NULL_CALIBRATION_MODE"
    --null-calibration-beta "$NULL_CALIBRATION_BETA"
    --null-question-text "$NULL_QUESTION_TEXT"
    --batch-size "$BATCH_SIZE"
    --max-new-tokens "$MAX_NEW_TOKENS"
  )

  if [[ "$SHUFFLE_OPTIONS" == "1" ]]; then
    CMD+=(--shuffle-options --option-shuffle-seed "$OPTION_SHUFFLE_SEED")
  fi

  if [[ "$OMIT_OPTION_LABELS" == "1" ]]; then
    CMD+=(--omit-option-labels)
  fi

  if [[ "$EXACT_OPTION_TEXT_INSTRUCTION" == "1" ]]; then
    CMD+=(--exact-option-text-instruction)
  fi

  if [[ -n "$LENGTH_NORM_ALPHA" ]]; then
    CMD+=(--length-norm-alpha "$LENGTH_NORM_ALPHA")
  fi

  if [[ "$ADD_PROMPT_BOS" == "1" ]]; then
    CMD+=(--add-prompt-bos)
  fi

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
