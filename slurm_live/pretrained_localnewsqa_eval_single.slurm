#!/bin/bash
#SBATCH --job-name=lnqa-pretrain
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48GB
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/amukher6/logs/slurm_logs/lnqa_pretrained.%j.out
#SBATCH --error=/scratch/amukher6/logs/slurm_logs/lnqa_pretrained.%j.err

set -euo pipefail

: "${MODEL_TYPE:?MODEL_TYPE must be set (custom|llama3_chat)}"
: "${META_TAG:?META_TAG must be set (with_metadata|without_metadata)}"

PYTHON=/home/amukher6/nanotron-env/bin/python
SCRIPT=/scratch/amukher6/metacul/src/14_sft_eval.py
DATASET=${DATASET:-iamshnoo/LocalNewsQA}
SPLIT=${SPLIT:-train}
LOCALE_ROLE=${LOCALE_ROLE:-target}
OUT_DIR=${OUT_DIR:-/scratch/amukher6/metacul/results/downstream_localnewsqa_pretrained}
RUN_TAG=${RUN_TAG:-pretrained}
CUSTOM_MODEL_PATH_WITH=${CUSTOM_MODEL_PATH_WITH:-/scratch/amukher6/metacul/models/combined_with_metadata_1b}
CUSTOM_MODEL_PATH_WITHOUT=${CUSTOM_MODEL_PATH_WITHOUT:-/scratch/amukher6/metacul/models/combined_without_metadata_1b}
TRAIN_META_TAG=${TRAIN_META_TAG:-$META_TAG}
EVAL_META_TAG=${EVAL_META_TAG:-$META_TAG}
LLAMA_MODEL_NAME=${LLAMA_MODEL_NAME:-meta-llama/Llama-3.2-1B-Instruct}
SHUFFLE_OPTIONS=${SHUFFLE_OPTIONS:-0}
OPTION_SHUFFLE_SEED=${OPTION_SHUFFLE_SEED:-42}
OMIT_OPTION_LABELS=${OMIT_OPTION_LABELS:-0}
EXACT_OPTION_TEXT_INSTRUCTION=${EXACT_OPTION_TEXT_INSTRUCTION:-0}
MCQ_SCORING=${MCQ_SCORING:-option_text_avg}
LENGTH_NORM_ALPHA=${LENGTH_NORM_ALPHA:-}
ANSWER_FORMAT=${ANSWER_FORMAT:-option}
MAX_EXAMPLES=${MAX_EXAMPLES:-}
SAMPLE_SEED=${SAMPLE_SEED:-42}
mkdir -p "$OUT_DIR"

OUT_JSONL="$OUT_DIR/localnewsqa_eval_${LOCALE_ROLE}_${EVAL_META_TAG}_${MODEL_TYPE}_${RUN_TAG}_c0.jsonl"

CMD=(
  "$PYTHON" "$SCRIPT"
  --model-type "$MODEL_TYPE"
  --dataset "$DATASET"
  --split "$SPLIT"
  --locale-role "$LOCALE_ROLE"
  --base-url "www.globalfactcheck.org"
  --output-jsonl "$OUT_JSONL"
  --url-corruption-rate 0
  --url-corruption-seed 42
  --mcq-scoring "$MCQ_SCORING"
  --answer-format "$ANSWER_FORMAT"
  --batch-size 2
  --max-new-tokens 64
  --temperature 0.6
  --top-p 0.9
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

if [[ -n "$MAX_EXAMPLES" ]]; then
  CMD+=(--max-examples "$MAX_EXAMPLES" --sample-seed "$SAMPLE_SEED")
fi

if [[ "$MODEL_TYPE" == "custom" ]]; then
  if [[ "$TRAIN_META_TAG" == "with_metadata" ]]; then
    CMD+=(--custom-model-path "$CUSTOM_MODEL_PATH_WITH" --metadata)
  else
    CMD+=(--custom-model-path "$CUSTOM_MODEL_PATH_WITHOUT")
  fi
  if [[ "$EVAL_META_TAG" == "without_metadata" ]]; then
    FILTERED_CMD=()
    for arg in "${CMD[@]}"; do
      if [[ "$arg" != "--metadata" ]]; then
        FILTERED_CMD+=("$arg")
      fi
    done
    CMD=("${FILTERED_CMD[@]}")
  fi
  if [[ "$EVAL_META_TAG" == "with_metadata" ]]; then
    case " ${CMD[*]} " in
      *" --metadata "*) ;;
      *) CMD+=(--metadata) ;;
    esac
  fi
else
  CMD+=(--llama-model-name "$LLAMA_MODEL_NAME")
  if [[ "$EVAL_META_TAG" == "with_metadata" ]]; then
    CMD+=(--metadata)
  fi
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
echo "Done: $OUT_JSONL"
