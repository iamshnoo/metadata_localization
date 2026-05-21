#!/bin/bash
#SBATCH --job-name=ext-packed
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48GB
#SBATCH --time=1-00:00:00
#SBATCH --output=/path/to/logs/slurm_logs/ext_packed.%j.out
#SBATCH --error=/path/to/logs/slurm_logs/ext_packed.%j.err

set -euo pipefail

: "${BENCHMARKS:?BENCHMARKS must be set (colon- or comma-separated list)}"

PYTHON=${PYTHON:-/path/to/nanotron-b200/bin/python}
SCRIPT=/path/to/metacul/src/25_sft_eval_external.py
OUT_ROOT=${OUT_ROOT:-/path/to/metacul/results/external_benchmarks_pretrained}
PAUSE_FILE=${PAUSE_FILE:-/path/to/metacul/slurm/.pause_external_gpuq_resume}
CUSTOM_MODEL_PATH_WITH=${CUSTOM_MODEL_PATH_WITH:-/path/to/metacul/models/combined_with_metadata_1b}
CUSTOM_MODEL_PATH_WITHOUT=${CUSTOM_MODEL_PATH_WITHOUT:-/path/to/metacul/models/combined_without_metadata_1b}
LLAMA_MODEL_NAME=${LLAMA_MODEL_NAME:-}
EVAL_SEED=${EVAL_SEED:-42}
OUT_SEED_TAG=${OUT_SEED_TAG:-seed_${EVAL_SEED}}
VARIANTS=${VARIANTS:-custom_tplus_eplus,custom_tminus_eminus}
SHUFFLE_OPTIONS=${SHUFFLE_OPTIONS:-0}
OPTION_SHUFFLE_SEED=${OPTION_SHUFFLE_SEED:-$EVAL_SEED}
METADATA_PROMPT_STYLE=${METADATA_PROMPT_STYLE:-legacy_code}
QA_PROMPT_STYLE=${QA_PROMPT_STYLE:-question}
ANSWER_CUE_STYLE=${ANSWER_CUE_STYLE:-none}
OMIT_OPTION_LABELS=${OMIT_OPTION_LABELS:-0}
EXACT_OPTION_TEXT_INSTRUCTION=${EXACT_OPTION_TEXT_INSTRUCTION:-0}
MCQ_SCORING=${MCQ_SCORING:-option_text_avg}
LENGTH_NORM_ALPHA=${LENGTH_NORM_ALPHA:-}
ADD_PROMPT_BOS=${ADD_PROMPT_BOS:-0}
NULL_CALIBRATION_MODE=${NULL_CALIBRATION_MODE:-none}
NULL_CALIBRATION_BETA=${NULL_CALIBRATION_BETA:-0.0}
NULL_QUESTION_TEXT=${NULL_QUESTION_TEXT:-[MASKED QUESTION]}
DISABLE_CUSTOM_CHAT_TEMPLATE=${DISABLE_CUSTOM_CHAT_TEMPLATE:-0}
RESUME_EVAL=${RESUME_EVAL:-1}

if [[ -f "$PAUSE_FILE" && "$OUT_ROOT" == *"/external_benchmarks_raw3b_unified_"* ]]; then
  echo "[paused] external raw3b unified packed eval disabled by $PAUSE_FILE for OUT_ROOT=$OUT_ROOT"
  exit 0
fi

BENCHMARKS_NORMALIZED=${BENCHMARKS//:/,}
VARIANTS_NORMALIZED=${VARIANTS//:/,}

IFS=',' read -r -a BENCHMARK_LIST <<< "$BENCHMARKS_NORMALIZED"

for BENCHMARK in "${BENCHMARK_LIST[@]}"; do
  BENCHMARK=$(echo "$BENCHMARK" | xargs)
  [[ -n "$BENCHMARK" ]] || continue
  OUT_DIR="$OUT_ROOT/$OUT_SEED_TAG/$BENCHMARK"
  mkdir -p "$OUT_DIR"

  CMD=(
    "$PYTHON" "$SCRIPT"
    --benchmark "$BENCHMARK"
    --variants "$VARIANTS_NORMALIZED"
    --batch-size 2
    --max-new-tokens 64
    --temperature 0.0
    --top-p 1.0
    --decoding greedy
    --seed "$EVAL_SEED"
    --output-dir "$OUT_DIR"
    --metadata-prompt-style "$METADATA_PROMPT_STYLE"
    --qa-prompt-style "$QA_PROMPT_STYLE"
    --answer-cue-style "$ANSWER_CUE_STYLE"
    --mcq-scoring "$MCQ_SCORING"
    --null-calibration-mode "$NULL_CALIBRATION_MODE"
    --null-calibration-beta "$NULL_CALIBRATION_BETA"
    --null-question-text "$NULL_QUESTION_TEXT"
    --custom-model-path-with-metadata "$CUSTOM_MODEL_PATH_WITH"
    --custom-model-path-without-metadata "$CUSTOM_MODEL_PATH_WITHOUT"
  )

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

  if [[ "$DISABLE_CUSTOM_CHAT_TEMPLATE" == "1" ]]; then
    CMD+=(--disable-custom-chat-template)
  fi

  if [[ "$RESUME_EVAL" == "1" ]]; then
    CMD+=(--resume)
  fi

  if [[ "$SHUFFLE_OPTIONS" == "1" ]]; then
    CMD+=(--shuffle-options --option-shuffle-seed "$OPTION_SHUFFLE_SEED")
  fi

  if [[ "$BENCHMARK" == "mmlu" ]]; then
    CMD+=(--dataset-config all)
  fi

  if [[ -n "$LLAMA_MODEL_NAME" ]]; then
    CMD+=(--llama-model-name "$LLAMA_MODEL_NAME")
  fi

  echo "Running [$BENCHMARK]: ${CMD[*]}"
  "${CMD[@]}"
  echo "Done [$BENCHMARK]: $OUT_DIR"
done
