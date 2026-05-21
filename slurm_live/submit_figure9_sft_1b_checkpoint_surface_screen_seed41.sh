#!/bin/bash
set -euo pipefail

CKPT="${1:-7764}"

ROOT=/path/to/metacul
SLURM="$ROOT/slurm/pretrained_localnewsqa_eval_single.slurm"
LOG_DIR=/path/to/logs/slurm_logs
TS=$(date +%Y%m%d_%H%M%S)
MANIFEST="$LOG_DIR/figure9_sft_1b_checkpoint_surface_screen_seed41_ckpt${CKPT}_${TS}.tsv"
RESULT_ROOT="$ROOT/results/downstream_localnewsqa_sft_1b_checkpoint_surface_screen_seed41/ckpt_${CKPT}"

WITH_BASE="$ROOT/models/combined_with_metadata_1b"
WITHOUT_BASE="$ROOT/models/combined_without_metadata_1b"
WITH_ADAPTER="$ROOT/models/sft/combined_with_metadata_sft_lora/checkpoint-${CKPT}"
WITHOUT_ADAPTER="$ROOT/models/sft/combined_without_metadata_sft_lora/checkpoint-${CKPT}"

mkdir -p "$LOG_DIR" "$RESULT_ROOT"
printf 'jobid\tconfig_slug\ttrain_meta\teval_meta\tout_dir\trun_tag\n' > "$MANIFEST"

declare -A CONFIG_EXPORTS

CONFIG_EXPORTS[countryfirst_qanswer_countryfinal_qmask025_bos]="METADATA_PROMPT_STYLE=country_first_strict,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.25"
CONFIG_EXPORTS[namegrounded_qanswer_countryfinal_qmask025_nobos]="METADATA_PROMPT_STYLE=name_grounded,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.25"
CONFIG_EXPORTS[nameplain_qanswer_countryfinal_qmask025_bos]="METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.25"
CONFIG_EXPORTS[codegrounded_question_countryfinal_labels_exact_nocal]="METADATA_PROMPT_STYLE=code_grounded,QA_PROMPT_STYLE=question,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=0,EXACT_OPTION_TEXT_INSTRUCTION=1,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=1.25,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0"
CONFIG_EXPORTS[codegrounded_question_final_labels_exact_nocal]="METADATA_PROMPT_STYLE=code_grounded,QA_PROMPT_STYLE=question,ANSWER_CUE_STYLE=final_answer_colon,OMIT_OPTION_LABELS=0,EXACT_OPTION_TEXT_INSTRUCTION=1,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=1.25,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0"
CONFIG_EXPORTS[namegrounded_question_countryfinal_labels_exact_nocal]="METADATA_PROMPT_STYLE=name_grounded,QA_PROMPT_STYLE=question,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=0,EXACT_OPTION_TEXT_INSTRUCTION=1,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=1.25,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0"
CONFIG_EXPORTS[legacycode_question_final_labels_exact_nocal]="METADATA_PROMPT_STYLE=legacy_code,QA_PROMPT_STYLE=question,ANSWER_CUE_STYLE=final_answer_colon,OMIT_OPTION_LABELS=0,EXACT_OPTION_TEXT_INSTRUCTION=1,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=1.25,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0"
CONFIG_EXPORTS[countryfirst_question_countryfinal_labels_exact_nocal]="METADATA_PROMPT_STYLE=country_first_strict,QA_PROMPT_STYLE=question,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=0,EXACT_OPTION_TEXT_INSTRUCTION=1,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=1.25,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0"

submit_one() {
  local config_slug="$1"
  local train_tag="$2"
  local eval_tag="$3"
  local train_short="$4"
  local eval_short="$5"
  local out_dir="$RESULT_ROOT/$config_slug"
  local run_tag="1b_ckpt${CKPT}_${config_slug}_${train_short}_${eval_short}_seed41"
  mkdir -p "$out_dir"
  local jobid
  jobid=$(sbatch --parsable \
    --export=ALL,MODEL_TYPE=custom,META_TAG="$eval_tag",TRAIN_META_TAG="$train_tag",EVAL_META_TAG="$eval_tag",BASE_MODEL_PATH_WITH="$WITH_BASE",BASE_MODEL_PATH_WITHOUT="$WITHOUT_BASE",PEFT_ADAPTER_PATH_WITH="$WITH_ADAPTER",PEFT_ADAPTER_PATH_WITHOUT="$WITHOUT_ADAPTER",OUT_DIR="$out_dir",RUN_TAG="$run_tag",SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED=41,SAMPLE_SEED=41,${CONFIG_EXPORTS[$config_slug]} \
    "$SLURM")
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$jobid" "$config_slug" "$train_tag" "$eval_tag" "$out_dir" "$run_tag" >> "$MANIFEST"
}

for config_slug in "${!CONFIG_EXPORTS[@]}"; do
  submit_one "$config_slug" with_metadata with_metadata tplus eplus
  submit_one "$config_slug" without_metadata without_metadata tminus eminus
done

echo "$MANIFEST"
