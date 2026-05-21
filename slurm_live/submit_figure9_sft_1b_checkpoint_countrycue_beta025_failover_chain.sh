#!/bin/bash
set -euo pipefail

CKPT="${1:-7764}"

ROOT=/path/to/metacul
SLURM_DIR="$ROOT/slurm"
PYTHON=/path/to/nanotron-b200/bin/python
RESULT_ROOT="$ROOT/results/downstream_localnewsqa_sft_figure9_1b_checkpoint_countrycue_beta025_rerun_multiseed/ckpt_${CKPT}"
ONE_B_PLOT_CSV="$ROOT/results/plots/plot8/plot_8_sft_target_split_multiseed_1b_checkpoint_countrycue_beta025_rerun.csv"
BASELINE_CSV="$ROOT/results/plots/plot8/plot_8_sft_target_split_multiseed.csv"
FAILOVER_SLURM="$SLURM_DIR/sft_1b_requirements_failover.slurm"
FINAL2500_CHAIN="$SLURM_DIR/submit_figure9_sft_1b_final2500_altprompt_chain.sh"

RESULT=$(
  RESULT_ROOT="$RESULT_ROOT" \
  RUN_ROOT_NAME=1b_chat_name_grounded_country_final_qanswer_nolabel_qmask025_bos \
  RUN_TAG_PREFIX=1b_chat_name_grounded_country_final_qanswer_nolabel_qmask025_bos \
  ONE_B_PLOT_CSV="$ONE_B_PLOT_CSV" \
  REQUIREMENT_ONLY=1 \
  OMIT_OPTION_LABELS=1 \
  EXACT_OPTION_TEXT_INSTRUCTION=0 \
  MCQ_SCORING=option_text_avg \
  METADATA_PROMPT_STYLE=name_grounded \
  QA_PROMPT_STYLE=question_answer \
  ANSWER_CUE_STYLE=country_final_answer_colon \
  ADD_PROMPT_BOS=1 \
  NULL_CALIBRATION_MODE=question_masked \
  NULL_CALIBRATION_BETA=0.25 \
  LENGTH_NORM_ALPHA=0.25 \
  bash "$SLURM_DIR/submit_figure9_sft_1b_checkpoint_multiseed.sh" "$CKPT"
)

echo "$RESULT"

SUMMARY_JOBID=$(printf '%s\n' "$RESULT" | awk -F= '/^summary_jobid=/{print $2}')
if [[ -z "$SUMMARY_JOBID" ]]; then
  echo "failed to parse summary_jobid from countrycue chain output" >&2
  exit 1
fi

FAILOVER_JOBID=$(sbatch --parsable \
  --dependency=afterok:${SUMMARY_JOBID} \
  --export=ALL,ONE_B_CSV="$ONE_B_PLOT_CSV",BASELINE_CSV="$BASELINE_CSV",FAILOVER_CHAIN="$FINAL2500_CHAIN",RESULT_ROOT=/path/to/metacul/results/downstream_localnewsqa_sft_figure9_1b_final2500_frozen1b_multiseed,ONE_B_PLOT_CSV=/path/to/metacul/results/plots/plot8/plot_8_sft_target_split_multiseed_1b_final2500_frozen1b.csv,RUN_ROOT_NAME=1b_codeg_labels_question_final,RUN_TAG_PREFIX=1b_codeg_labels_question_final,OMIT_OPTION_LABELS=0,EXACT_OPTION_TEXT_INSTRUCTION=1,MCQ_SCORING=option_text_avg,METADATA_PROMPT_STYLE=code_grounded,QA_PROMPT_STYLE=question,ANSWER_CUE_STYLE=final_answer_colon,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0,LENGTH_NORM_ALPHA=1.25,PYTHON="$PYTHON" \
  "$FAILOVER_SLURM")

echo "countrycue_failover_jobid=${FAILOVER_JOBID}"
