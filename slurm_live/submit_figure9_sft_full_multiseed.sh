#!/bin/bash
set -euo pipefail

SLURM=/path/to/metacul/slurm/pretrained_localnewsqa_eval_single.slurm
LOG_DIR=/path/to/logs/slurm_logs
RESULT_ROOT=/path/to/metacul/results/downstream_localnewsqa_sft_figure9_full_multiseed
TS=$(date +%Y%m%d_%H%M%S)
MANIFEST=$LOG_DIR/figure9_sft_full_multiseed_${TS}.tsv

mkdir -p "$LOG_DIR" "$RESULT_ROOT"
printf 'jobid\tfamily\tseed\ttrain_meta\teval_meta\tout_dir\trun_tag\n' > "$MANIFEST"

submit_family() {
  local family="$1"
  local seed="$2"
  local with_path="$3"
  local without_path="$4"
  local out_dir="$5"
  local run_tag_prefix="$6"
  local metadata_prompt_style="$7"
  local qa_prompt_style="$8"
  local answer_cue_style="$9"
  local omit_option_labels="${10}"
  local exact_option_text_instruction="${11}"
  local add_prompt_bos="${12}"
  local null_calibration_mode="${13}"
  local length_norm_alpha="${14}"
  local null_calibration_beta="${15}"

  mkdir -p "$out_dir"

  for combo in \
    "with_metadata with_metadata tplus eplus" \
    "with_metadata without_metadata tplus eminus" \
    "without_metadata with_metadata tminus eplus" \
    "without_metadata without_metadata tminus eminus"
  do
    read -r train_tag eval_tag train_short eval_short <<< "$combo"
    run_tag="${run_tag_prefix}_${family}_${train_short}_${eval_short}_seed${seed}"
    jobid=$(sbatch --parsable \
      --export=ALL,MODEL_TYPE=custom,META_TAG="$eval_tag",TRAIN_META_TAG="$train_tag",EVAL_META_TAG="$eval_tag",CUSTOM_MODEL_PATH_WITH="$with_path",CUSTOM_MODEL_PATH_WITHOUT="$without_path",OUT_DIR="$out_dir",RUN_TAG="$run_tag",SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED="$seed",OMIT_OPTION_LABELS="$omit_option_labels",EXACT_OPTION_TEXT_INSTRUCTION="$exact_option_text_instruction",MCQ_SCORING=option_text_avg,METADATA_PROMPT_STYLE="$metadata_prompt_style",QA_PROMPT_STYLE="$qa_prompt_style",ANSWER_CUE_STYLE="$answer_cue_style",SAMPLE_SEED="$seed",ADD_PROMPT_BOS="$add_prompt_bos",NULL_CALIBRATION_MODE="$null_calibration_mode",NULL_CALIBRATION_BETA="$null_calibration_beta",LENGTH_NORM_ALPHA="$length_norm_alpha" \
      "$SLURM")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$jobid" "$family" "$seed" "$train_tag" "$eval_tag" "$out_dir" "$run_tag" >> "$MANIFEST"
  done
}

for seed in 41 42 43 44 45; do
  submit_family \
    1b_chat \
    "$seed" \
    /path/to/metacul/models/sft/combined_with_metadata_chat \
    /path/to/metacul/models/sft/combined_without_metadata_chat \
    "$RESULT_ROOT/1b_chat_codeg_labels_question_final/seed_${seed}" \
    sftfig9 \
    code_grounded \
    question \
    final_answer_colon \
    0 \
    1 \
    0 \
    none \
    1.25 \
    0.0

  submit_family \
    3b_best3b_chat \
    "$seed" \
    /path/to/metacul/models/sft/combined_with_metadata_3b_best3b_chat \
    /path/to/metacul/models/sft/combined_without_metadata_3b_best3b_chat \
    "$RESULT_ROOT/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_${seed}" \
    sftfig9 \
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
