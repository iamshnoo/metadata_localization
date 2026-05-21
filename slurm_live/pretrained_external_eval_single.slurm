#!/bin/bash
#SBATCH --job-name=ext-pretrain
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48GB
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/path/to/logs/slurm_logs/ext_pretrained.%j.out
#SBATCH --error=/path/to/logs/slurm_logs/ext_pretrained.%j.err

set -euo pipefail

: "${BENCHMARK:?BENCHMARK must be set (geomlama|globalopinionqa|worldvaluebench|mmlu)}"
: "${VARIANT:?VARIANT must be set (e.g., custom_tplus_eplus)}"

ENV_FILE=${ENV_FILE:-}
if [[ -n "" && -f "" ]]; then
  set -a
  source ""
  set +a
fi

export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/path/to/cache}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/path/to/cache/pip}
export HF_HOME=${HF_HOME:-/path/to/cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/path/to/cache/huggingface/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/path/to/cache/huggingface/datasets}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/path/to/cache/huggingface}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HF_ENABLE_PARALLEL_LOADING=${HF_ENABLE_PARALLEL_LOADING:-false}
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

PYTHON=${PYTHON:-/path/to/nanotron-b200/bin/python}
SCRIPT=/path/to/metacul/src/25_sft_eval_external.py
OUT_ROOT=${OUT_ROOT:-/path/to/metacul/results/external_benchmarks_pretrained}
CUSTOM_MODEL_PATH_WITH=${CUSTOM_MODEL_PATH_WITH:-/path/to/metacul/models/combined_with_metadata_1b}
CUSTOM_MODEL_PATH_WITHOUT=${CUSTOM_MODEL_PATH_WITHOUT:-/path/to/metacul/models/combined_without_metadata_1b}
CUSTOM_TOKENIZER_PATH=${CUSTOM_TOKENIZER_PATH:-meta-llama/Llama-3.2-1B}
LLAMA_MODEL_NAME=${LLAMA_MODEL_NAME:-}
VARIANTS=${VARIANTS:-$VARIANT}
VARIANTS=${VARIANTS//:/,}
BATCH_SIZE=${BATCH_SIZE:-2}
EVAL_SEED=${EVAL_SEED:-42}
SHUFFLE_OPTIONS=${SHUFFLE_OPTIONS:-0}
OPTION_SHUFFLE_SEED=${OPTION_SHUFFLE_SEED:-$EVAL_SEED}
MAX_SAMPLES=${MAX_SAMPLES:-0}
METADATA_PROMPT_STYLE=${METADATA_PROMPT_STYLE:-legacy_code}
QA_PROMPT_STYLE=${QA_PROMPT_STYLE:-question}
ANSWER_CUE_STYLE=${ANSWER_CUE_STYLE:-none}
OMIT_OPTION_LABELS=${OMIT_OPTION_LABELS:-0}
EXACT_OPTION_TEXT_INSTRUCTION=${EXACT_OPTION_TEXT_INSTRUCTION:-0}
MCQ_SCORING=${MCQ_SCORING:-option_text_avg}
LENGTH_NORM_ALPHA=${LENGTH_NORM_ALPHA:-}
ADD_PROMPT_BOS=${ADD_PROMPT_BOS:-0}
METADATA_TAG_MODE=${METADATA_TAG_MODE:-correct}
NULL_CALIBRATION_MODE=${NULL_CALIBRATION_MODE:-none}
NULL_CALIBRATION_BETA=${NULL_CALIBRATION_BETA:-0.0}
NULL_QUESTION_TEXT=${NULL_QUESTION_TEXT:-[MASKED QUESTION]}
DISABLE_CUSTOM_CHAT_TEMPLATE=${DISABLE_CUSTOM_CHAT_TEMPLATE:-0}
FORCE_CUSTOM_CHAT_PROMPT=${FORCE_CUSTOM_CHAT_PROMPT:-0}
RESUME_EVAL=${RESUME_EVAL:-1}
FLAT_OUTPUT=${FLAT_OUTPUT:-0}
OUT_SEED_TAG=${OUT_SEED_TAG:-seed_${EVAL_SEED}}
if [[ "$FLAT_OUTPUT" == "1" ]]; then
  OUT_DIR="$OUT_ROOT/$OUT_SEED_TAG/$BENCHMARK"
else
  OUT_DIR="$OUT_ROOT/$OUT_SEED_TAG/$BENCHMARK/$VARIANT"
fi
mkdir -p "$OUT_DIR"

CMD=(
  "$PYTHON" "$SCRIPT"
  --benchmark "$BENCHMARK"
  --variants "$VARIANTS"
  --batch-size "$BATCH_SIZE"
  --max-new-tokens 64
  --temperature 0.0
  --top-p 1.0
  --decoding greedy
  --seed "$EVAL_SEED"
  --output-dir "$OUT_DIR"
  --metadata-tag-mode "$METADATA_TAG_MODE"
  --metadata-prompt-style "$METADATA_PROMPT_STYLE"
  --qa-prompt-style "$QA_PROMPT_STYLE"
  --answer-cue-style "$ANSWER_CUE_STYLE"
  --mcq-scoring "$MCQ_SCORING"
  --null-calibration-mode "$NULL_CALIBRATION_MODE"
  --null-calibration-beta "$NULL_CALIBRATION_BETA"
  --null-question-text "$NULL_QUESTION_TEXT"
  --custom-model-path-with-metadata "$CUSTOM_MODEL_PATH_WITH"
  --custom-model-path-without-metadata "$CUSTOM_MODEL_PATH_WITHOUT"
  --custom-tokenizer-path "$CUSTOM_TOKENIZER_PATH"
)

if [[ "$MAX_SAMPLES" != "0" ]]; then
  CMD+=(--max-samples "$MAX_SAMPLES")
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

if [[ "$DISABLE_CUSTOM_CHAT_TEMPLATE" == "1" ]]; then
  CMD+=(--disable-custom-chat-template)
fi

if [[ "$FORCE_CUSTOM_CHAT_PROMPT" == "1" ]]; then
  CMD+=(--force-custom-chat-prompt)
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

echo "Running: ${CMD[*]}"
"$PYTHON" - <<'PY'
import os
import transformers, tokenizers, huggingface_hub
print(f"transformers={transformers.__version__}")
print(f"tokenizers={tokenizers.__version__}")
print(f"huggingface_hub={huggingface_hub.__version__}")
print(f"hf_token_present={bool(os.getenv('HF_TOKEN'))}")
PY
"${CMD[@]}"
echo "Done: $OUT_DIR"
