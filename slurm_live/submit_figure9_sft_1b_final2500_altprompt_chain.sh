#!/bin/bash
set -euo pipefail

SLURM_DIR=/path/to/metacul/slurm
NAME_SUFFIX=${NAME_SUFFIX:-_final2500}
MAX_STEPS=${MAX_STEPS:-2500}
SAVE_STEPS=${SAVE_STEPS:-500}
EVAL_STEPS=${EVAL_STEPS:-500}
WITH_TRAIN_JOB_NAME=${WITH_TRAIN_JOB_NAME:-}
WITHOUT_TRAIN_JOB_NAME=${WITHOUT_TRAIN_JOB_NAME:-}
WITH_MERGE_JOB_NAME=${WITH_MERGE_JOB_NAME:-}
WITHOUT_MERGE_JOB_NAME=${WITHOUT_MERGE_JOB_NAME:-}
WITH_MODEL_PATH=${WITH_MODEL_PATH:-/path/to/metacul/models/sft/combined_with_metadata_1b${NAME_SUFFIX}_chat}
WITHOUT_MODEL_PATH=${WITHOUT_MODEL_PATH:-/path/to/metacul/models/sft/combined_without_metadata_1b${NAME_SUFFIX}_chat}
RESULT_ROOT=${RESULT_ROOT:-/path/to/metacul/results/downstream_localnewsqa_sft_figure9_1b${NAME_SUFFIX}_altprompt_multiseed}
ONE_B_PLOT_CSV=${ONE_B_PLOT_CSV:-/path/to/metacul/results/plots/plot8/plot_8_sft_target_split_multiseed_1b${NAME_SUFFIX}_altprompt.csv}

with_train_args=(--parsable)
without_train_args=(--parsable)
with_merge_args=(--parsable)
without_merge_args=(--parsable)

if [[ -n "$WITH_TRAIN_JOB_NAME" ]]; then
  with_train_args+=(--job-name="$WITH_TRAIN_JOB_NAME")
fi
if [[ -n "$WITHOUT_TRAIN_JOB_NAME" ]]; then
  without_train_args+=(--job-name="$WITHOUT_TRAIN_JOB_NAME")
fi
if [[ -n "$WITH_MERGE_JOB_NAME" ]]; then
  with_merge_args+=(--job-name="$WITH_MERGE_JOB_NAME")
fi
if [[ -n "$WITHOUT_MERGE_JOB_NAME" ]]; then
  without_merge_args+=(--job-name="$WITHOUT_MERGE_JOB_NAME")
fi

WITH_TRAIN_JOB=$(sbatch "${with_train_args[@]}" \
  --export=ALL,NAME_SUFFIX="$NAME_SUFFIX",MAX_STEPS="$MAX_STEPS",SAVE_STEPS="$SAVE_STEPS",EVAL_STEPS="$EVAL_STEPS" \
  "$SLURM_DIR/run_sft_1b_final2500_with_metadata.sbatch")
WITHOUT_TRAIN_JOB=$(sbatch "${without_train_args[@]}" \
  --export=ALL,NAME_SUFFIX="$NAME_SUFFIX",MAX_STEPS="$MAX_STEPS",SAVE_STEPS="$SAVE_STEPS",EVAL_STEPS="$EVAL_STEPS" \
  "$SLURM_DIR/run_sft_1b_final2500_without_metadata.sbatch")

WITH_MERGE_JOB=$(sbatch "${with_merge_args[@]}" \
  --dependency=afterok:${WITH_TRAIN_JOB} \
  --export=ALL,NAME_SUFFIX="$NAME_SUFFIX" \
  "$SLURM_DIR/merge_sft_1b_final2500.sbatch" with)
WITHOUT_MERGE_JOB=$(sbatch "${without_merge_args[@]}" \
  --dependency=afterok:${WITHOUT_TRAIN_JOB} \
  --export=ALL,NAME_SUFFIX="$NAME_SUFFIX" \
  "$SLURM_DIR/merge_sft_1b_final2500.sbatch" without)

echo "with_train_jobid=${WITH_TRAIN_JOB}"
echo "without_train_jobid=${WITHOUT_TRAIN_JOB}"
echo "with_merge_jobid=${WITH_MERGE_JOB}"
echo "without_merge_jobid=${WITHOUT_MERGE_JOB}"

RESULT=$(WITH_MODEL_PATH="$WITH_MODEL_PATH" \
  WITHOUT_MODEL_PATH="$WITHOUT_MODEL_PATH" \
  RESULT_ROOT="$RESULT_ROOT" \
  ONE_B_PLOT_CSV="$ONE_B_PLOT_CSV" \
  SLURM_SCRIPT="${SLURM_SCRIPT:-}" \
  SUMMARY_JOB_NAME="${SUMMARY_JOB_NAME:-}" \
  RUN_ROOT_NAME="${RUN_ROOT_NAME:-}" \
  RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-}" \
  OMIT_OPTION_LABELS="${OMIT_OPTION_LABELS:-}" \
  EXACT_OPTION_TEXT_INSTRUCTION="${EXACT_OPTION_TEXT_INSTRUCTION:-}" \
  MCQ_SCORING="${MCQ_SCORING:-}" \
  METADATA_PROMPT_STYLE="${METADATA_PROMPT_STYLE:-}" \
  QA_PROMPT_STYLE="${QA_PROMPT_STYLE:-}" \
  ANSWER_CUE_STYLE="${ANSWER_CUE_STYLE:-}" \
  ADD_PROMPT_BOS="${ADD_PROMPT_BOS:-}" \
  NULL_CALIBRATION_MODE="${NULL_CALIBRATION_MODE:-}" \
  NULL_CALIBRATION_BETA="${NULL_CALIBRATION_BETA:-}" \
  LENGTH_NORM_ALPHA="${LENGTH_NORM_ALPHA:-}" \
  REQUIREMENT_ONLY=1 \
  WITH_DEPENDENCY_JOBID="$WITH_MERGE_JOB" \
  WITHOUT_DEPENDENCY_JOBID="$WITHOUT_MERGE_JOB" \
  bash "$SLURM_DIR/submit_figure9_sft_1b_altprompt_multiseed.sh")

echo "$RESULT"
