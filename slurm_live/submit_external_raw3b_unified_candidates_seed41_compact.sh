#!/bin/bash
set -euo pipefail

ROOT=/path/to/metacul
SLURM="${SLURM_SCRIPT:-$ROOT/slurm/pretrained_external_eval_packed.slurm}"
LOG_DIR=/path/to/logs/slurm_logs
TS=$(date +%Y%m%d_%H%M%S)
MANIFEST="$LOG_DIR/external_raw3b_unified_candidates_seed41_compact_${TS}.tsv"

mkdir -p "$LOG_DIR"
printf 'jobid\tconfig_slug\tout_root\n' > "$MANIFEST"

BENCHMARKS=geomlama:globalopinionqa:worldvaluebench:mmlu
VARIANTS=custom_tplus_eplus:custom_tminus_eminus
SEED=41
PYTHON=/path/to/nanotron-b200/bin/python

submit_config_job() {
  local config_slug="$1"
  local export_suffix="$2"
  local out_root="$RESULT_ROOT/$config_slug"
  local export_str="ALL,PYTHON=${PYTHON},BENCHMARKS=${BENCHMARKS},VARIANTS=${VARIANTS},OUT_ROOT=${out_root},EVAL_SEED=${SEED},SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED=${SEED},CUSTOM_MODEL_PATH_WITH=${ROOT}/models/combined_with_metadata_3b,CUSTOM_MODEL_PATH_WITHOUT=${ROOT}/models/combined_without_metadata_3b,${export_suffix}"
  local jobid
  jobid=$(sbatch --parsable --export="$export_str" "$SLURM")
  printf '%s\t%s\t%s\n' "$jobid" "$config_slug" "$out_root" >> "$MANIFEST"
}

submit_config_job \
  "name_plain_qanswer_countryfinal_qmask025_nobos" \
  "METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.25"

submit_config_job \
  "name_plain_question_countryfinal_letter_qmask05_bos" \
  "METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=0,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_letter,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.5"

submit_config_job \
  "name_plain_qanswer_countryfinal_letter_qmask025_nobos" \
  "METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=0,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_letter,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.25"

echo "$MANIFEST"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/results/external_benchmarks_raw3b_unified_candidates_seed41_compact}"
