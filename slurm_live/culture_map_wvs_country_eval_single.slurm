#!/bin/bash
#SBATCH --job-name=wvs-country
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48GB
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/amukher6/logs/slurm_logs/wvs_country.%j.out
#SBATCH --error=/scratch/amukher6/logs/slurm_logs/wvs_country.%j.err

set -euo pipefail

: "${VARIANT:?VARIANT must be set (e.g. maple_1b_tplus_eplus)}"

export XDG_CACHE_HOME=/scratch/amukher6/cache
export PIP_CACHE_DIR=/scratch/amukher6/cache/pip
export HF_HOME=/scratch/amukher6/cache/huggingface
export HF_HUB_CACHE=/scratch/amukher6/cache/huggingface/hub
export HF_DATASETS_CACHE=/scratch/amukher6/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/scratch/amukher6/cache/huggingface
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HF_ENABLE_PARALLEL_LOADING=${HF_ENABLE_PARALLEL_LOADING:-false}
mkdir -p \
  "$XDG_CACHE_HOME" \
  "$PIP_CACHE_DIR" \
  "$HF_HOME" \
  "$HF_HUB_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$TRANSFORMERS_CACHE"

PYTHON=/home/amukher6/nanotron-env/bin/python
export PYTHONPATH=/scratch/amukher6/culture-map/src${PYTHONPATH:+:$PYTHONPATH}

DATA_DIR=${DATA_DIR:-/scratch/amukher6/culture-map/data/paper_osf}
OUT_DIR=${OUT_DIR:-/scratch/amukher6/metacul/results/culture_map_wvs_country_eval}
ENV_FILE=${ENV_FILE:-/scratch/amukher6/culture-map/.agents/.env}
BASE_URL=${BASE_URL:-https://www.worldvaluessurvey.org/WVSContents.jsp?CMSID=tradrat}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
OVERWRITE=${OVERWRITE:-0}
COUNTRY_CODES=${COUNTRY_CODES:-}
mkdir -p "$OUT_DIR"

CMD=(
  "$PYTHON" -m culture_map.cli run-local-country-eval
  --data-dir "$DATA_DIR"
  --output-dir "$OUT_DIR"
  --env-file "$ENV_FILE"
  --base-url "$BASE_URL"
  --variant "$VARIANT"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --max-new-tokens "$MAX_NEW_TOKENS"
)

if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi

if [[ -n "$COUNTRY_CODES" ]]; then
  IFS=',' read -r -a COUNTRY_CODE_ARRAY <<< "$COUNTRY_CODES"
  for code in "${COUNTRY_CODE_ARRAY[@]}"; do
    if [[ -n "$code" ]]; then
      CMD+=(--country-code "$code")
    fi
  done
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
echo "Done: VARIANT=$VARIANT OUT_DIR=$OUT_DIR"
