#!/bin/bash
set -euo pipefail

CKPT="${1:-7764}"

ROOT=/path/to/metacul
SLURM_DIR="$ROOT/slurm"
PYTHON=/path/to/nanotron-b200/bin/python
RESULT_ROOT="$ROOT/results/downstream_localnewsqa_sft_figure9_1b_checkpoint_frozen1b_rerun_multiseed/ckpt_${CKPT}"
ONE_B_PLOT_CSV="$ROOT/results/plots/plot8/plot_8_sft_target_split_multiseed_1b_checkpoint_frozen1b_rerun.csv"
BASELINE_CSV="$ROOT/results/plots/plot8/plot_8_sft_target_split_multiseed.csv"
FAILOVER_SLURM="$SLURM_DIR/sft_1b_requirements_failover.slurm"
COUNTRYCUE_CHAIN="$SLURM_DIR/submit_figure9_sft_1b_checkpoint_countrycue_beta025_failover_chain.sh"

RESULT=$(
  RESULT_ROOT="$RESULT_ROOT" \
  RUN_ROOT_NAME=1b_codeg_labels_question_final \
  RUN_TAG_PREFIX=1b_codeg_labels_question_final \
  ONE_B_PLOT_CSV="$ONE_B_PLOT_CSV" \
  REQUIREMENT_ONLY=1 \
  OMIT_OPTION_LABELS=0 \
  EXACT_OPTION_TEXT_INSTRUCTION=1 \
  MCQ_SCORING=option_text_avg \
  METADATA_PROMPT_STYLE=code_grounded \
  QA_PROMPT_STYLE=question \
  ANSWER_CUE_STYLE=final_answer_colon \
  ADD_PROMPT_BOS=0 \
  NULL_CALIBRATION_MODE=none \
  NULL_CALIBRATION_BETA=0.0 \
  LENGTH_NORM_ALPHA=1.25 \
  bash "$SLURM_DIR/submit_figure9_sft_1b_checkpoint_multiseed.sh" "$CKPT"
)

echo "$RESULT"

SUMMARY_JOBID=$(printf '%s\n' "$RESULT" | awk -F= '/^summary_jobid=/{print $2}')
if [[ -z "$SUMMARY_JOBID" ]]; then
  echo "failed to parse summary_jobid from frozen1b chain output" >&2
  exit 1
fi

FAILOVER_JOBID=$(sbatch --parsable \
  --dependency=afterok:${SUMMARY_JOBID} \
  --export=ALL,ONE_B_CSV="$ONE_B_PLOT_CSV",BASELINE_CSV="$BASELINE_CSV",FAILOVER_CHAIN="$COUNTRYCUE_CHAIN",PYTHON="$PYTHON" \
  "$FAILOVER_SLURM")

echo "frozen1b_failover_jobid=${FAILOVER_JOBID}"
