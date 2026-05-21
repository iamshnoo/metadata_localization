#!/bin/bash
set -euo pipefail

SLURM=/path/to/metacul/slurm/pretrained_localnewsqa_adversarial_grid.slurm
LOG_DIR=/path/to/logs/slurm_logs
RESULT_ROOT=/path/to/metacul/results/downstream_localnewsqa_pretrained_adversarial_figure9_urlmask
TS=$(date +%Y%m%d_%H%M%S)
MANIFEST=$LOG_DIR/figure9_aligned_adversarial_urlmask_${TS}.tsv

mkdir -p "$LOG_DIR" "$RESULT_ROOT"
printf 'jobid\tfamily\tmeta_tag\tout_dir\trun_tag\trate_list\tmode\n' > "$MANIFEST"

submit_job() {
  local family="$1"
  local meta_tag="$2"
  local with_path="$3"
  local without_path="$4"
  local run_tag="$5"
  local metadata_prompt_style="$6"
  local qa_prompt_style="$7"
  local answer_cue_style="$8"
  local omit_option_labels="$9"
  local exact_option_text_instruction="${10}"
  local add_prompt_bos="${11}"
  local null_calibration_mode="${12}"
  local length_norm_alpha="${13}"
  local null_calibration_beta="${14}"
  local rate_list="${15}"

  local out_dir="$RESULT_ROOT/${family}"
  mkdir -p "$out_dir"

  jobid=$(sbatch --parsable \
    --export=ALL,MODEL_TYPE=custom,META_TAG="$meta_tag",CUSTOM_MODEL_PATH_WITH="$with_path",CUSTOM_MODEL_PATH_WITHOUT="$without_path",OUT_DIR="$out_dir",RUN_TAG="$run_tag",RATE_LIST="$rate_list",URL_CORRUPTION_MODE=url_country_mask,URL_MASK_TOKEN=xx,SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED=42,OMIT_OPTION_LABELS="$omit_option_labels",EXACT_OPTION_TEXT_INSTRUCTION="$exact_option_text_instruction",MCQ_SCORING=option_text_avg,METADATA_PROMPT_STYLE="$metadata_prompt_style",QA_PROMPT_STYLE="$qa_prompt_style",ANSWER_CUE_STYLE="$answer_cue_style",ADD_PROMPT_BOS="$add_prompt_bos",NULL_CALIBRATION_MODE="$null_calibration_mode",NULL_CALIBRATION_BETA="$null_calibration_beta",LENGTH_NORM_ALPHA="$length_norm_alpha" \
    "$SLURM")

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$jobid" "$family" "$meta_tag" "$out_dir" "$run_tag" "$rate_list" "url_country_mask" >> "$MANIFEST"
}

submit_job \
  1b \
  with_metadata \
  /path/to/metacul/models/combined_with_metadata_1b \
  /path/to/metacul/models/combined_without_metadata_1b \
  1b_tplus_urlmask \
  code_grounded \
  question \
  final_answer_colon \
  0 \
  1 \
  0 \
  none \
  1.25 \
  0.0 \
  '0;0.25;0.5;0.75;1.0'

submit_job \
  1b \
  without_metadata \
  /path/to/metacul/models/combined_with_metadata_1b \
  /path/to/metacul/models/combined_without_metadata_1b \
  1b_tminus_urlmask \
  code_grounded \
  question \
  final_answer_colon \
  0 \
  1 \
  0 \
  none \
  1.25 \
  0.0 \
  '0'

submit_job \
  3b \
  with_metadata \
  /path/to/metacul/models/combined_with_metadata_3b \
  /path/to/metacul/models/combined_without_metadata_3b \
  3b_tplus_urlmask \
  name_grounded \
  question_answer \
  final_answer_colon \
  1 \
  0 \
  1 \
  question_masked \
  0.25 \
  0.5 \
  '0;0.25;0.5;0.75;1.0'

submit_job \
  3b \
  without_metadata \
  /path/to/metacul/models/combined_with_metadata_3b \
  /path/to/metacul/models/combined_without_metadata_3b \
  3b_tminus_urlmask \
  name_grounded \
  question_answer \
  final_answer_colon \
  1 \
  0 \
  1 \
  question_masked \
  0.25 \
  0.5 \
  '0'

echo "$MANIFEST"
