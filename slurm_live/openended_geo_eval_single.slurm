#!/bin/bash
#SBATCH --job-name=opengeo
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48GB
#SBATCH --time=1-00:00:00
#SBATCH --output=/path/to/logs/slurm_logs/openended_geo.%j.out
#SBATCH --error=/path/to/logs/slurm_logs/openended_geo.%j.err

set -euo pipefail

: "${BENCHMARK:?BENCHMARK must be set}"
: "${VARIANT:?VARIANT must be set}"

PYTHON=${PYTHON:-/path/to/nanotron-env/bin/python}
SCRIPT=/path/to/metacul/src/62_openended_geo_eval.py
OUT_ROOT=${OUT_ROOT:-/path/to/metacul/results/openended_geo_eval}
INPUT_PATH=${INPUT_PATH:-}
MULTINATIVQA_CONFIG=${MULTINATIVQA_CONFIG:-english_qa}
SPLIT=${SPLIT:-}
MAX_SAMPLES=${MAX_SAMPLES:-0}
EVAL_SEED=${EVAL_SEED:-42}
CUSTOM_MODEL_PATH_WITH=${CUSTOM_MODEL_PATH_WITH:-/path/to/metacul/models/sft/combined_with_metadata_chat}
CUSTOM_MODEL_PATH_WITHOUT=${CUSTOM_MODEL_PATH_WITHOUT:-/path/to/metacul/models/sft/combined_without_metadata_chat}
HF_MODEL_NAME=${HF_MODEL_NAME:-meta-llama/Llama-3.2-1B-Instruct}
PROMPT_STYLE=${PROMPT_STYLE:-croq}
METADATA_PROMPT_STYLE=${METADATA_PROMPT_STYLE:-name_grounded}
BASE_URL=${BASE_URL:-www.globalfactcheck.org}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-96}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
DECODING=${DECODING:-greedy}
ENV_PATH=${ENV_PATH:-}
JUDGE_MODEL=${JUDGE_MODEL:-gpt-4o-mini}
JUDGE_MAX_OUTPUT_TOKENS=${JUDGE_MAX_OUTPUT_TOKENS:-64}
JUDGE_SLEEP_SECONDS=${JUDGE_SLEEP_SECONDS:-0.0}
RESUME_EVAL=${RESUME_EVAL:-1}
OUT_DIR="$OUT_ROOT/${BENCHMARK}/${VARIANT}"
mkdir -p "$OUT_DIR"

CMD=(
  "$PYTHON" "$SCRIPT"
  --benchmark "$BENCHMARK"
  --variants "$VARIANT"
  --multinativqa-config "$MULTINATIVQA_CONFIG"
  --seed "$EVAL_SEED"
  --output-dir "$OUT_DIR"
  --custom-model-path-with-metadata "$CUSTOM_MODEL_PATH_WITH"
  --custom-model-path-without-metadata "$CUSTOM_MODEL_PATH_WITHOUT"
  --hf-model-name "$HF_MODEL_NAME"
  --prompt-style "$PROMPT_STYLE"
  --metadata-prompt-style "$METADATA_PROMPT_STYLE"
  --base-url "$BASE_URL"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --decoding "$DECODING"
  --env-path "$ENV_PATH"
  --judge-model "$JUDGE_MODEL"
  --judge-max-output-tokens "$JUDGE_MAX_OUTPUT_TOKENS"
  --judge-sleep-seconds "$JUDGE_SLEEP_SECONDS"
)

if [[ -n "$INPUT_PATH" ]]; then
  CMD+=(--input-path "$INPUT_PATH")
fi

if [[ -n "$SPLIT" ]]; then
  CMD+=(--split "$SPLIT")
fi

if [[ "$MAX_SAMPLES" != "0" ]]; then
  CMD+=(--max-samples "$MAX_SAMPLES")
fi

if [[ "$RESUME_EVAL" == "1" ]]; then
  CMD+=(--resume)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
echo "Done: $OUT_DIR"
