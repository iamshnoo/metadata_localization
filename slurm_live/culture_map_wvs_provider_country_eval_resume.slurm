#!/bin/bash
#SBATCH --job-name=wvs-provider
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=contrib
#SBATCH --time=3-00:00:00
#SBATCH --output=/path/to/metacul/logs/slurm_logs/%x-%j.out
#SBATCH --error=/path/to/metacul/logs/slurm_logs/%x-%j.err

set -euo pipefail

: "${PROVIDER:?PROVIDER must be set}"
: "${MODEL:?MODEL must be set}"
: "${OUT_DIR:?OUT_DIR must be set}"
: "${COUNTRY_CODES:?COUNTRY_CODES must be set}"

PYTHON_BIN=${PYTHON_BIN:-/path/to/nanotron-b200/bin/python}
DATA_DIR=${DATA_DIR:-/path/to/culture-map/data/paper_osf}
ENV_FILE=${ENV_FILE:-}
PROMPT_BASE_URL=${PROMPT_BASE_URL:-https://www.worldvaluessurvey.org/WVSContents.jsp?CMSID=tradrat}
COUNTRY_SCOPE=${COUNTRY_SCOPE:-all_human}
METADATA_PROMPT_STYLE=${METADATA_PROMPT_STYLE:-name_typical}
OPTION_STYLE=${OPTION_STYLE:-long_score}
TEMPERATURE=${TEMPERATURE:-0.0}
DELAY=${DELAY:-0.0}
API_BASE_URL=${API_BASE_URL:-}
OVERWRITE=${OVERWRITE:-0}

export PYTHONPATH=/path/to/culture-map/src${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$OUT_DIR" /path/to/metacul/logs/slurm_logs

CMD=(
  "$PYTHON_BIN" -m culture_map.cli run-provider-country-eval
  --provider "$PROVIDER"
  --data-dir "$DATA_DIR"
  --output-dir "$OUT_DIR"
  --env-file "$ENV_FILE"
  --prompt-base-url "$PROMPT_BASE_URL"
  --model "$MODEL"
  --country-scope "$COUNTRY_SCOPE"
  --metadata-prompt-style "$METADATA_PROMPT_STYLE"
  --option-style "$OPTION_STYLE"
  --temperature "$TEMPERATURE"
  --delay "$DELAY"
)

if [[ -n "$API_BASE_URL" ]]; then
  CMD+=(--api-base-url "$API_BASE_URL")
fi

if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi

IFS=',' read -r -a COUNTRY_CODE_ARRAY <<< "$COUNTRY_CODES"
for code in "${COUNTRY_CODE_ARRAY[@]}"; do
  if [[ -n "$code" ]]; then
    CMD+=(--country-code "$code")
  fi
done

echo "Running: ${CMD[*]}"
"${CMD[@]}"
echo "Done: PROVIDER=$PROVIDER MODEL=$MODEL OUT_DIR=$OUT_DIR"
