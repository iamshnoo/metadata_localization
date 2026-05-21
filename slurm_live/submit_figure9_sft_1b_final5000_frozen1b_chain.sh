#!/bin/bash
set -euo pipefail

SLURM_DIR=/path/to/metacul/slurm

NAME_SUFFIX=${NAME_SUFFIX:-_final5000}
MAX_STEPS=${MAX_STEPS:-5000}
SAVE_STEPS=${SAVE_STEPS:-500}
EVAL_STEPS=${EVAL_STEPS:-500}
RESULT_ROOT=${RESULT_ROOT:-/path/to/metacul/results/downstream_localnewsqa_sft_figure9_1b_final5000_frozen1b_multiseed}
ONE_B_PLOT_CSV=${ONE_B_PLOT_CSV:-/path/to/metacul/results/plots/plot8/plot_8_sft_target_split_multiseed_1b_final5000_frozen1b.csv}

NAME_SUFFIX="$NAME_SUFFIX" \
MAX_STEPS="$MAX_STEPS" \
SAVE_STEPS="$SAVE_STEPS" \
EVAL_STEPS="$EVAL_STEPS" \
RESULT_ROOT="$RESULT_ROOT" \
ONE_B_PLOT_CSV="$ONE_B_PLOT_CSV" \
RUN_ROOT_NAME=1b_codeg_labels_question_final \
RUN_TAG_PREFIX=1b_codeg_labels_question_final \
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
bash "$SLURM_DIR/submit_figure9_sft_1b_final2500_altprompt_chain.sh"
