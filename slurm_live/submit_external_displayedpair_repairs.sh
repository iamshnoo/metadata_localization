#!/bin/bash
set -euo pipefail

ROOT=/path/to/metacul
SLURM="$ROOT/slurm/pretrained_external_eval_single.slurm"
LOG_DIR=/path/to/logs/slurm_logs
TS=$(date +%Y%m%d_%H%M%S)
MANIFEST=$LOG_DIR/external_displayedpair_repairs_${TS}.tsv

mkdir -p "$LOG_DIR"
printf 'jobid\tfamily\tbenchmark\tvariant\tseed\tconfig_slug\tout_root\n' > "$MANIFEST"

VARIANTS=(custom_tplus_eplus custom_tminus_eminus)

has_summary() {
  local dir="$1"
  compgen -G "${dir}/*summary.csv" > /dev/null || compgen -G "${dir}/*summary.json" > /dev/null
}

submit_if_missing() {
  local family="$1"
  local benchmark="$2"
  local variant="$3"
  local seed="$4"
  local config_slug="$5"
  local out_root="$6"
  local export_suffix="$7"
  local custom_with="$8"
  local custom_without="$9"

  local out_dir="${out_root}/seed_${seed}/${benchmark}/${variant}"
  mkdir -p "$out_dir"
  if has_summary "$out_dir"; then
    return
  fi

  local export_str="ALL,BENCHMARK=${benchmark},VARIANT=${variant},OUT_ROOT=${out_root},EVAL_SEED=${seed},SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED=${seed},CUSTOM_MODEL_PATH_WITH=${custom_with},CUSTOM_MODEL_PATH_WITHOUT=${custom_without},${export_suffix}"
  local jobid
  jobid=$(sbatch --parsable --export="$export_str" "$SLURM")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$jobid" "$family" "$benchmark" "$variant" "$seed" "$config_slug" "$out_root" >> "$MANIFEST"
}

raw1b_config_slug=code_grounded_question_final_a125
raw1b_export_suffix="METADATA_PROMPT_STYLE=code_grounded,QA_PROMPT_STYLE=question,ANSWER_CUE_STYLE=final_answer_colon,OMIT_OPTION_LABELS=0,EXACT_OPTION_TEXT_INSTRUCTION=1,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=1.25,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0"
raw1b_out_root="$ROOT/results/external_benchmarks_tuned_multiseed_followup/raw_1b/${raw1b_config_slug}"

raw3b_geomwvb_config_slug=name_plain_qanswer_countryfinal_a025_nobos
raw3b_geomwvb_export_suffix="METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0"
raw3b_geomwvb_out_root="$ROOT/results/external_benchmarks_tuned_multiseed_followup/raw_3b/${raw3b_geomwvb_config_slug}"

raw3b_goqa_config_slug=name_strict_qanswer_nolabel_qmask05
raw3b_goqa_export_suffix="METADATA_PROMPT_STYLE=name_strict,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.5"
raw3b_goqa_out_root="$ROOT/results/external_benchmarks_tuned_multiseed_followup/raw_3b/${raw3b_goqa_config_slug}"

raw3b_mmlu_config_slug=legacy_defaults_shuffle
raw3b_mmlu_export_suffix="METADATA_PROMPT_STYLE=legacy_code,QA_PROMPT_STYLE=question,ANSWER_CUE_STYLE=none,OMIT_OPTION_LABELS=0,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0"
raw3b_mmlu_out_root="$ROOT/results/external_benchmarks_mmlu_followup/raw_3b/${raw3b_mmlu_config_slug}"

for seed in 42 43 44 45; do
  for variant in "${VARIANTS[@]}"; do
    for benchmark in geomlama globalopinionqa worldvaluebench mmlu; do
      submit_if_missing \
        raw_1b \
        "$benchmark" \
        "$variant" \
        "$seed" \
        "$raw1b_config_slug" \
        "$raw1b_out_root" \
        "$raw1b_export_suffix" \
        "$ROOT/models/combined_with_metadata_1b" \
        "$ROOT/models/combined_without_metadata_1b"
    done

    for benchmark in geomlama worldvaluebench; do
      submit_if_missing \
        raw_3b \
        "$benchmark" \
        "$variant" \
        "$seed" \
        "$raw3b_geomwvb_config_slug" \
        "$raw3b_geomwvb_out_root" \
        "$raw3b_geomwvb_export_suffix" \
        "$ROOT/models/combined_with_metadata_3b" \
        "$ROOT/models/combined_without_metadata_3b"
    done

    submit_if_missing \
      raw_3b \
      globalopinionqa \
      "$variant" \
      "$seed" \
      "$raw3b_goqa_config_slug" \
      "$raw3b_goqa_out_root" \
      "$raw3b_goqa_export_suffix" \
      "$ROOT/models/combined_with_metadata_3b" \
      "$ROOT/models/combined_without_metadata_3b"

    submit_if_missing \
      raw_3b \
      mmlu \
      "$variant" \
      "$seed" \
      "$raw3b_mmlu_config_slug" \
      "$raw3b_mmlu_out_root" \
      "$raw3b_mmlu_export_suffix" \
      "$ROOT/models/combined_with_metadata_3b" \
      "$ROOT/models/combined_without_metadata_3b"
  done
done

echo "$MANIFEST"
