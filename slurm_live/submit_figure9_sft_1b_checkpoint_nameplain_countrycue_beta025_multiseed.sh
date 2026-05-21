#!/bin/bash
set -euo pipefail

CKPT="${1:-7764}"

ROOT=/path/to/metacul
SLURM_DIR="$ROOT/slurm"
SLURM_SCRIPT="${SLURM_SCRIPT:-$SLURM_DIR/pretrained_localnewsqa_eval_single_gpuq_sftfull.slurm}"

RESULT_ROOT=${RESULT_ROOT:-$ROOT/results/downstream_localnewsqa_sft_figure9_1b_checkpoint_nameplain_countrycue_beta025_multiseed/ckpt_${CKPT}}
ONE_B_PLOT_CSV=${ONE_B_PLOT_CSV:-$ROOT/results/plots/plot8/plot_8_sft_target_split_multiseed_1b_checkpoint_nameplain_countrycue_beta025.csv}

RESULT=$(
  SLURM_SCRIPT="$SLURM_SCRIPT" \
  RESULT_ROOT="$RESULT_ROOT" \
  RUN_ROOT_NAME=1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos \
  RUN_TAG_PREFIX=1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos \
  ONE_B_PLOT_CSV="$ONE_B_PLOT_CSV" \
  REQUIREMENT_ONLY=1 \
  OMIT_OPTION_LABELS=1 \
  EXACT_OPTION_TEXT_INSTRUCTION=0 \
  MCQ_SCORING=option_text_avg \
  METADATA_PROMPT_STYLE=name_plain \
  QA_PROMPT_STYLE=question_answer \
  ANSWER_CUE_STYLE=country_final_answer_colon \
  ADD_PROMPT_BOS=1 \
  NULL_CALIBRATION_MODE=question_masked \
  NULL_CALIBRATION_BETA=0.25 \
  LENGTH_NORM_ALPHA=0.25 \
  RESUME=1 \
  bash "$SLURM_DIR/submit_figure9_sft_1b_checkpoint_multiseed.sh" "$CKPT"
)

echo "$RESULT"
