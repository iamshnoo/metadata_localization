#!/bin/bash
set -euo pipefail

ROOT=/path/to/metacul
SLURM="$ROOT/slurm/pretrained_external_eval_single.slurm"
LOG_DIR=/path/to/logs/slurm_logs
TS=$(date +%Y%m%d_%H%M%S)
MANIFEST=$LOG_DIR/external_raw3b_geomlama_focus2_${TS}.tsv

mkdir -p "$LOG_DIR"
printf 'jobid\tbenchmark\tvariant\tconfig_slug\tout_root\n' > "$MANIFEST"

BENCHMARK=geomlama
VARIANTS=(custom_tplus_eplus custom_tminus_eminus)
SEED=41

CONFIGS=(
  "name_plain_qanswer_countryfinal_a025_bos|METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0"
  "name_plain_qanswer_countrycorrect_a025_bos|METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_the_correct_answer_is,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0"
  "name_plain_qanswer_countryanswer_a025_bos|METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0"
  "name_plain_qanswer_countryfinal_a0_bos|METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.0,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0"
  "name_plain_qanswer_countryfinal_a025_nobos|METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0"
  "name_plain_qanswer_final_a0_nobos|METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.0,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0"
  "name_plain_question_countryfinal_a025_bos|METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0"
  "name_plain_question_countrycorrect_a025_bos|METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question,ANSWER_CUE_STYLE=country_the_correct_answer_is,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0"
  "name_plain_qanswer_countryfinal_sum_a025_bos|METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_sum,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0"
  "name_plain_qanswer_countryfinal_qmasknometa_a025|METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=question_masked_no_metadata,NULL_CALIBRATION_BETA=0.25"
  "country_first_qanswer_countryfinal_a025_bos|METADATA_PROMPT_STYLE=country_first_strict,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0"
  "country_first_qanswer_countrycorrect_a025_bos|METADATA_PROMPT_STYLE=country_first_strict,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_the_correct_answer_is,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0"
  "country_first_question_countryfinal_a025_bos|METADATA_PROMPT_STYLE=country_first_strict,QA_PROMPT_STYLE=question,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0"
  "code_grounded_qanswer_countryfinal_a025_bos|METADATA_PROMPT_STYLE=code_grounded,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0"
  "code_grounded_qanswer_countrycorrect_a025_bos|METADATA_PROMPT_STYLE=code_grounded,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_the_correct_answer_is,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0"
  "name_strict_qanswer_countryfinal_a025_bos|METADATA_PROMPT_STYLE=name_strict,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0"
  "name_grounded_qanswer_countryfinal_a025_bos|METADATA_PROMPT_STYLE=name_grounded,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0"
)

submit_job() {
  local variant="$1"
  local config_slug="$2"
  local export_suffix="$3"
  local out_root="$ROOT/results/external_benchmarks_raw3b_geomlama_focus2/$config_slug"
  local export_str="ALL,BENCHMARK=${BENCHMARK},VARIANT=${variant},OUT_ROOT=${out_root},EVAL_SEED=${SEED},SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED=${SEED},CUSTOM_MODEL_PATH_WITH=${ROOT}/models/combined_with_metadata_3b,CUSTOM_MODEL_PATH_WITHOUT=${ROOT}/models/combined_without_metadata_3b,${export_suffix}"
  local jobid
  jobid=$(sbatch --parsable --export="$export_str" "$SLURM")
  printf '%s\t%s\t%s\t%s\t%s\n' "$jobid" "$BENCHMARK" "$variant" "$config_slug" "$out_root" >> "$MANIFEST"
}

for variant in "${VARIANTS[@]}"; do
  for config in "${CONFIGS[@]}"; do
    slug=${config%%|*}
    export_suffix=${config#*|}
    submit_job "$variant" "$slug" "$export_suffix"
  done
done

echo "$MANIFEST"
