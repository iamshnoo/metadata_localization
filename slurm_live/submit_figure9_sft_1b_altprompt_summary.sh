#!/bin/bash
set -euo pipefail

MANIFEST="${1:?usage: $0 <altprompt-manifest.tsv> [sft_root] [one_b_plot_csv] }"
SFT_ROOT="${2:-/path/to/metacul/results/downstream_localnewsqa_sft_figure9_1b_altprompt_multiseed}"
ONE_B_PLOT_CSV="${3:-/path/to/metacul/results/plots/plot8/plot_8_sft_target_split_multiseed_1b_altprompt.csv}"
PYTHON=${PYTHON:-/path/to/nanotron-b200/bin/python}
LOG_DIR=/path/to/logs/slurm_logs
SUMMARY_SCRIPT=/path/to/metacul/src/57_localnewsqa_sft_multiseed_summary_1b_alt.py
CHECK_SCRIPT=/path/to/metacul/src/59_sft_requirements_check.py
SUMMARY_JOB_NAME=${SUMMARY_JOB_NAME:-sft1b-altsum}

mkdir -p "$LOG_DIR"

JOB_IDS=$(tail -n +2 "$MANIFEST" | cut -f1 | paste -sd: -)
if [[ -z "$JOB_IDS" ]]; then
  echo "no job IDs found in manifest: $MANIFEST" >&2
  exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
JOB_ID=$(sbatch --parsable \
  --job-name="$SUMMARY_JOB_NAME" \
  --partition=interactive \
  --mem=8G \
  --time=02:00:00 \
  --dependency=afterok:${JOB_IDS} \
  --output="$LOG_DIR/sft1b_alt_sum.%j.out" \
  --error="$LOG_DIR/sft1b_alt_sum.%j.err" \
  --wrap="set -euo pipefail; $PYTHON $SUMMARY_SCRIPT --sft-root $SFT_ROOT --plot-csv $ONE_B_PLOT_CSV; $PYTHON $CHECK_SCRIPT --one-b-csv $ONE_B_PLOT_CSV")

echo "$JOB_ID"
