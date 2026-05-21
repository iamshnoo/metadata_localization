#!/bin/bash
#SBATCH --job-name=sft1breq
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48GB
#SBATCH --time=1-00:00:00
#SBATCH --output=/path/to/logs/slurm_logs/sft1breq.%j.out
#SBATCH --error=/path/to/logs/slurm_logs/sft1breq.%j.err

set -euo pipefail

: "${MODEL_TYPE:?MODEL_TYPE must be set (custom|llama3_chat)}"
: "${META_TAG:?META_TAG must be set (with_metadata|without_metadata)}"

PYTHON=${PYTHON:-/path/to/nanotron-b200/bin/python}
SCRIPT=/path/to/metacul/src/14_sft_eval.py
DATASET=${DATASET:-YOUR_HF_USERNAME/LocalNewsQA}
SPLIT=${SPLIT:-train}
LOCALE_ROLE=${LOCALE_ROLE:-target}
OUT_DIR=${OUT_DIR:-/path/to/metacul/results/downstream_localnewsqa_pretrained}
RUN_TAG=${RUN_TAG:-pretrained}
CUSTOM_MODEL_PATH_WITH=${CUSTOM_MODEL_PATH_WITH:-/path/to/metacul/models/combined_with_metadata_1b}
CUSTOM_MODEL_PATH_WITHOUT=${CUSTOM_MODEL_PATH_WITHOUT:-/path/to/metacul/models/combined_without_metadata_1b}
BASE_MODEL_PATH_WITH=${BASE_MODEL_PATH_WITH:-/path/to/metacul/models/combined_with_metadata_1b}
BASE_MODEL_PATH_WITHOUT=${BASE_MODEL_PATH_WITHOUT:-/path/to/metacul/models/combined_without_metadata_1b}
PEFT_ADAPTER_PATH_WITH=${PEFT_ADAPTER_PATH_WITH:-}
PEFT_ADAPTER_PATH_WITHOUT=${PEFT_ADAPTER_PATH_WITHOUT:-}
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
METADATA_PROMPT_STYLE=${METADATA_PROMPT_STYLE:-legacy_code}
QA_PROMPT_STYLE=${QA_PROMPT_STYLE:-question}
ANSWER_CUE_STYLE=${ANSWER_CUE_STYLE:-none}
MAX_EXAMPLES=${MAX_EXAMPLES:-}
SAMPLE_SEED=${SAMPLE_SEED:-42}
ADD_PROMPT_BOS=${ADD_PROMPT_BOS:-0}
NULL_CALIBRATION_MODE=${NULL_CALIBRATION_MODE:-none}
NULL_CALIBRATION_BETA=${NULL_CALIBRATION_BETA:-1.0}
NULL_QUESTION_TEXT=${NULL_QUESTION_TEXT:-N/A}
RESUME=${RESUME:-0}

export HF_HOME=${HF_HOME:-/path/to/cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
mkdir -p "$OUT_DIR"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

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
  --metadata-prompt-style "$METADATA_PROMPT_STYLE"
  --qa-prompt-style "$QA_PROMPT_STYLE"
  --answer-cue-style "$ANSWER_CUE_STYLE"
  --null-calibration-mode "$NULL_CALIBRATION_MODE"
  --null-calibration-beta "$NULL_CALIBRATION_BETA"
  --null-question-text "$NULL_QUESTION_TEXT"
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

if [[ "$ADD_PROMPT_BOS" == "1" ]]; then
  CMD+=(--add-prompt-bos)
fi

if [[ "$RESUME" == "1" ]]; then
  CMD+=(--resume)
fi

if [[ "$MODEL_TYPE" == "custom" ]]; then
  if [[ "$TRAIN_META_TAG" == "with_metadata" ]]; then
    if [[ -n "$PEFT_ADAPTER_PATH_WITH" ]]; then
      CMD+=(--base-model-path "$BASE_MODEL_PATH_WITH" --peft-adapter-path "$PEFT_ADAPTER_PATH_WITH" --metadata)
    else
      CMD+=(--custom-model-path "$CUSTOM_MODEL_PATH_WITH" --metadata)
    fi
  else
    if [[ -n "$PEFT_ADAPTER_PATH_WITHOUT" ]]; then
      CMD+=(--base-model-path "$BASE_MODEL_PATH_WITHOUT" --peft-adapter-path "$PEFT_ADAPTER_PATH_WITHOUT")
    else
      CMD+=(--custom-model-path "$CUSTOM_MODEL_PATH_WITHOUT")
    fi
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
