#!/bin/bash
#SBATCH --job-name=lnqa-base
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48GB
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/path/to/logs/slurm_logs/lnqa_baseline.%j.out
#SBATCH --error=/path/to/logs/slurm_logs/lnqa_baseline.%j.err

set -euo pipefail

: "${MODEL_NAME:?MODEL_NAME must be set}"
: "${MODEL_SLUG:?MODEL_SLUG must be set}"
: "${META_TAG:?META_TAG must be set (with_metadata|without_metadata)}"

ENV_FILE=${ENV_FILE:-}
if [[ -n "" && -f "" ]]; then
  set -a
  source ""
  set +a
fi

export XDG_CACHE_HOME=/path/to/cache
export PIP_CACHE_DIR=/path/to/cache/pip
export HF_HOME=/path/to/cache/huggingface
export HF_HUB_CACHE=/path/to/cache/huggingface/hub
export HF_DATASETS_CACHE=/path/to/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/path/to/cache/huggingface
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HF_ENABLE_PARALLEL_LOADING=${HF_ENABLE_PARALLEL_LOADING:-false}
mkdir -p \
  "$XDG_CACHE_HOME" \
  "$PIP_CACHE_DIR" \
  "$HF_HOME" \
  "$HF_HUB_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$TRANSFORMERS_CACHE"

PYTHON=${PYTHON:-/path/to/nanotron-env/bin/python}
SCRIPT=/path/to/metacul/src/14_sft_eval.py
DATASET=${DATASET:-YOUR_HF_USERNAME/LocalNewsQA}
SPLIT=${SPLIT:-train}
LOCALE_ROLE=${LOCALE_ROLE:-target}
OUT_DIR=${OUT_DIR:-/path/to/metacul/results/downstream_localnewsqa_external_baselines}
BATCH_SIZE=${BATCH_SIZE:-2}
RUN_TAG=${RUN_TAG:-}
SHUFFLE_OPTIONS=${SHUFFLE_OPTIONS:-0}
OPTION_SHUFFLE_SEED=${OPTION_SHUFFLE_SEED:-42}
MCQ_SCORING=${MCQ_SCORING:-option_text_avg}
ANSWER_FORMAT=${ANSWER_FORMAT:-option}
LOCALIZE_MODEL=${LOCALIZE_MODEL:-0}
SPLIT_TYPE_FILTER=${SPLIT_TYPE_FILTER:-all}
mkdir -p "$OUT_DIR"

OUT_JSONL="$OUT_DIR/localnewsqa_${MODEL_SLUG}_${META_TAG}_${LOCALE_ROLE}.jsonl"
if [[ -n "$RUN_TAG" ]]; then
  OUT_JSONL="$OUT_DIR/localnewsqa_${MODEL_SLUG}_${META_TAG}_${LOCALE_ROLE}_${RUN_TAG}.jsonl"
fi

echo "MODEL_NAME=$MODEL_NAME"
echo "MODEL_SLUG=$MODEL_SLUG"
echo "META_TAG=$META_TAG"
echo "OUT_JSONL=$OUT_JSONL"

MODEL_SOURCE="$MODEL_NAME"
if [[ "$LOCALIZE_MODEL" == "1" && "$MODEL_NAME" != /* ]]; then
  : "${HF_TOKEN:?HF_TOKEN must be set when LOCALIZE_MODEL=1 for a remote model}"
  LOCAL_TMP_ROOT="${SLURM_TMPDIR:-/tmp}"
  LOCAL_JOB_SUFFIX="${SLURM_JOB_ID:-$$}"
  LOCAL_MODEL_DIR="$LOCAL_TMP_ROOT/${MODEL_SLUG}_snapshot_${LOCAL_JOB_SUFFIX}"
  export MODEL_NAME HF_TOKEN LOCAL_MODEL_DIR
  echo "LOCALIZE_MODEL=1"
  echo "LOCAL_MODEL_DIR=$LOCAL_MODEL_DIR"
  "$PYTHON" - <<'PY'
import os
import shutil
from huggingface_hub import snapshot_download

repo_id = os.environ["MODEL_NAME"]
token = os.environ["HF_TOKEN"]
local_dir = os.environ["LOCAL_MODEL_DIR"]

snapshot_path = snapshot_download(
    repo_id=repo_id,
    token=token,
    cache_dir=os.environ.get("HF_HUB_CACHE"),
)

if os.path.exists(local_dir):
    shutil.rmtree(local_dir)

shutil.copytree(snapshot_path, local_dir, symlinks=False)
print(f"[INFO] Localized model snapshot to {local_dir}")
PY
  MODEL_SOURCE="$LOCAL_MODEL_DIR"
fi

"$PYTHON" - <<'PY'
import transformers, tokenizers, huggingface_hub
print(f"transformers={transformers.__version__}")
print(f"tokenizers={tokenizers.__version__}")
print(f"huggingface_hub={huggingface_hub.__version__}")
PY

CMD=(
  "$PYTHON" "$SCRIPT"
  --model-type llama3_chat
  --llama-model-name "$MODEL_SOURCE"
  --dataset "$DATASET"
  --split "$SPLIT"
  --locale-role "$LOCALE_ROLE"
  --split-type-filter "$SPLIT_TYPE_FILTER"
  --base-url "www.globalfactcheck.org"
  --output-jsonl "$OUT_JSONL"
  --url-corruption-rate 0
  --url-corruption-seed 42
  --mcq-scoring "$MCQ_SCORING"
  --answer-format "$ANSWER_FORMAT"
  --batch-size "$BATCH_SIZE"
  --max-new-tokens 64
  --temperature 0.6
  --top-p 0.9
)

if [[ "$SHUFFLE_OPTIONS" == "1" ]]; then
  CMD+=(--shuffle-options --option-shuffle-seed "$OPTION_SHUFFLE_SEED")
fi

if [[ "$META_TAG" == "with_metadata" ]]; then
  CMD+=(--metadata)
fi

CMD+=(--resume)

echo "Running: ${CMD[*]}"
"${CMD[@]}"
echo "Done: $OUT_JSONL"
