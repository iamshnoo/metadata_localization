#!/bin/bash
set -euo pipefail

ROOT=/path/to/metacul
SLURM="$ROOT/slurm/pretrained_external_eval_single.slurm"
LOG_DIR=/path/to/logs/slurm_logs
TS=$(date +%Y%m%d_%H%M%S)
MANIFEST=$LOG_DIR/external_mmlu_followup_${TS}.tsv

mkdir -p "$LOG_DIR"
printf 'jobid\tfamily\tvariant\tseed\tconfig_slug\tout_root\n' > "$MANIFEST"

SEEDS=(42 43 44 45)
VARIANTS=(custom_tplus_eplus custom_tminus_eminus)
BENCHMARK=mmlu

submit_job() {
  local family="$1"
  local variant="$2"
  local seed="$3"
  local config_slug="$4"
  local out_root="$5"
  local export_suffix="$6"
  local custom_with="$7"
  local custom_without="$8"

  local export_str="ALL,BENCHMARK=${BENCHMARK},VARIANT=${variant},OUT_ROOT=${out_root},EVAL_SEED=${seed},SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED=${seed},CUSTOM_MODEL_PATH_WITH=${custom_with},CUSTOM_MODEL_PATH_WITHOUT=${custom_without},${export_suffix}"
  local jobid
  jobid=$(sbatch --parsable --export="$export_str" "$SLURM")
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$jobid" "$family" "$variant" "$seed" "$config_slug" "$out_root" >> "$MANIFEST"
}

raw1b_config_slug=code_grounded_question_final_a125
raw1b_export_suffix="METADATA_PROMPT_STYLE=code_grounded,QA_PROMPT_STYLE=question,ANSWER_CUE_STYLE=final_answer_colon,OMIT_OPTION_LABELS=0,EXACT_OPTION_TEXT_INSTRUCTION=1,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=1.25,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0"
raw1b_out_root="$ROOT/results/external_benchmarks_mmlu_followup/raw_1b/${raw1b_config_slug}"

raw3b_config_slug=legacy_defaults_shuffle
raw3b_export_suffix="METADATA_PROMPT_STYLE=legacy_code,QA_PROMPT_STYLE=question,ANSWER_CUE_STYLE=none,OMIT_OPTION_LABELS=0,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0"
raw3b_out_root="$ROOT/results/external_benchmarks_mmlu_followup/raw_3b/${raw3b_config_slug}"

for seed in "${SEEDS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    submit_job \
      raw_1b \
      "$variant" \
      "$seed" \
      "$raw1b_config_slug" \
      "$raw1b_out_root" \
      "$raw1b_export_suffix" \
      "$ROOT/models/combined_with_metadata_1b" \
      "$ROOT/models/combined_without_metadata_1b"

    submit_job \
      raw_3b \
      "$variant" \
      "$seed" \
      "$raw3b_config_slug" \
      "$raw3b_out_root" \
      "$raw3b_export_suffix" \
      "$ROOT/models/combined_with_metadata_3b" \
      "$ROOT/models/combined_without_metadata_3b"
  done
done

echo "$MANIFEST"
