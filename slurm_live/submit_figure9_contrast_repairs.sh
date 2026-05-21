#!/bin/bash
set -euo pipefail

SLURM=/path/to/metacul/slurm/pretrained_localnewsqa_eval_single.slurm
LOG_DIR=/path/to/logs/slurm_logs
RESULT_ROOT=/path/to/metacul/results/downstream_localnewsqa_pretrained_figure9_contrast_full_multiseed
TS=$(date +%Y%m%d_%H%M%S)
MANIFEST=$LOG_DIR/figure9_contrast_repairs_${TS}.tsv

mkdir -p "$LOG_DIR" "$RESULT_ROOT"
printf 'jobid\tfamily\tseed\ttrain_meta\teval_meta\tlocale_role\tout_dir\trun_tag\n' > "$MANIFEST"

submit_variant() {
  local family="$1"
  local seed="$2"
  local train_tag="$3"
  local eval_tag="$4"
  local with_path="$5"
  local without_path="$6"
  local out_dir="$7"
  local run_tag="$8"
  local metadata_prompt_style="$9"
  local qa_prompt_style="${10}"
  local answer_cue_style="${11}"
  local omit_option_labels="${12}"
  local exact_option_text_instruction="${13}"
  local add_prompt_bos="${14}"
  local null_calibration_mode="${15}"
  local length_norm_alpha="${16}"
  local null_calibration_beta="${17}"

  mkdir -p "$out_dir"

  local jobid
  jobid=$(sbatch --parsable \
    --export=ALL,MODEL_TYPE=custom,LOCALE_ROLE=contrast,META_TAG="$eval_tag",TRAIN_META_TAG="$train_tag",EVAL_META_TAG="$eval_tag",CUSTOM_MODEL_PATH_WITH="$with_path",CUSTOM_MODEL_PATH_WITHOUT="$without_path",OUT_DIR="$out_dir",RUN_TAG="$run_tag",SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED="$seed",OMIT_OPTION_LABELS="$omit_option_labels",EXACT_OPTION_TEXT_INSTRUCTION="$exact_option_text_instruction",MCQ_SCORING=option_text_avg,METADATA_PROMPT_STYLE="$metadata_prompt_style",QA_PROMPT_STYLE="$qa_prompt_style",ANSWER_CUE_STYLE="$answer_cue_style",SAMPLE_SEED="$seed",ADD_PROMPT_BOS="$add_prompt_bos",NULL_CALIBRATION_MODE="$null_calibration_mode",NULL_CALIBRATION_BETA="$null_calibration_beta",LENGTH_NORM_ALPHA="$length_norm_alpha" \
    "$SLURM")

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$jobid" "$family" "$seed" "$train_tag" "$eval_tag" "contrast" "$out_dir" "$run_tag" >> "$MANIFEST"
}

# 1B repairs: seed 44 was truncated; seed 45 never started.
for seed in 44 45; do
  submit_variant \
    1b \
    "$seed" \
    with_metadata \
    with_metadata \
    /path/to/metacul/models/combined_with_metadata_1b \
    /path/to/metacul/models/combined_without_metadata_1b \
    "$RESULT_ROOT/1b_codeg_labels_question_final/seed_${seed}" \
    frozenfig9contrast_1b_tplus_eplus_seed${seed} \
    code_grounded \
    question \
    final_answer_colon \
    0 \
    1 \
    0 \
    none \
    1.25 \
    0.0

  submit_variant \
    1b \
    "$seed" \
    without_metadata \
    without_metadata \
    /path/to/metacul/models/combined_with_metadata_1b \
    /path/to/metacul/models/combined_without_metadata_1b \
    "$RESULT_ROOT/1b_codeg_labels_question_final/seed_${seed}" \
    frozenfig9contrast_1b_tminus_eminus_seed${seed} \
    code_grounded \
    question \
    final_answer_colon \
    0 \
    1 \
    0 \
    none \
    1.25 \
    0.0
done

# 3B repairs: seeds 42-44 were truncated; seed 45 never started.
for seed in 42 43 44 45; do
  submit_variant \
    3b \
    "$seed" \
    with_metadata \
    with_metadata \
    /path/to/metacul/models/combined_with_metadata_3b \
    /path/to/metacul/models/combined_without_metadata_3b \
    "$RESULT_ROOT/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_${seed}" \
    frozenfig9contrast_3b_tplus_eplus_seed${seed} \
    name_grounded \
    question_answer \
    final_answer_colon \
    1 \
    0 \
    1 \
    question_masked \
    0.25 \
    0.5

  submit_variant \
    3b \
    "$seed" \
    without_metadata \
    without_metadata \
    /path/to/metacul/models/combined_with_metadata_3b \
    /path/to/metacul/models/combined_without_metadata_3b \
    "$RESULT_ROOT/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_${seed}" \
    frozenfig9contrast_3b_tminus_eminus_seed${seed} \
    name_grounded \
    question_answer \
    final_answer_colon \
    1 \
    0 \
    1 \
    question_masked \
    0.25 \
    0.5
done

echo "$MANIFEST"
