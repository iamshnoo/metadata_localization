#!/bin/bash
set -euo pipefail

SLURM=/path/to/metacul/slurm/pretrained_localnewsqa_eval_single.slurm
LOG_DIR=/path/to/logs/slurm_logs
RESULT_ROOT=/path/to/metacul/results/downstream_localnewsqa_sft_1b_checkpoint_screen
TS=$(date +%Y%m%d_%H%M%S)
MANIFEST=$LOG_DIR/figure9_sft_1b_checkpoint_screen_${TS}.tsv

WITH_BASE=/path/to/metacul/models/combined_with_metadata_1b
WITHOUT_BASE=/path/to/metacul/models/combined_without_metadata_1b
WITH_ADAPTER_ROOT=/path/to/metacul/models/sft/combined_with_metadata_sft_lora
WITHOUT_ADAPTER_ROOT=/path/to/metacul/models/sft/combined_without_metadata_sft_lora

mkdir -p "$LOG_DIR" "$RESULT_ROOT"
printf 'jobid\tcheckpoint\ttrain_meta\teval_meta\tout_dir\trun_tag\n' > "$MANIFEST"

submit_one() {
  local ckpt="$1"
  local train_tag="$2"
  local eval_tag="$3"
  local run_tag="$4"
  local out_dir="$RESULT_ROOT/checkpoint_${ckpt}"
  local with_adapter="$WITH_ADAPTER_ROOT/checkpoint-${ckpt}"
  local without_adapter="$WITHOUT_ADAPTER_ROOT/checkpoint-${ckpt}"
  mkdir -p "$out_dir"

  jobid=$(sbatch --parsable \
    --export=ALL,MODEL_TYPE=custom,META_TAG="$eval_tag",TRAIN_META_TAG="$train_tag",EVAL_META_TAG="$eval_tag",BASE_MODEL_PATH_WITH="$WITH_BASE",BASE_MODEL_PATH_WITHOUT="$WITHOUT_BASE",PEFT_ADAPTER_PATH_WITH="$with_adapter",PEFT_ADAPTER_PATH_WITHOUT="$without_adapter",OUT_DIR="$out_dir",RUN_TAG="$run_tag",SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED=41,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,METADATA_PROMPT_STYLE=name_grounded,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=final_answer_colon,SAMPLE_SEED=41,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.5,LENGTH_NORM_ALPHA=0.25 \
    "$SLURM")
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$jobid" "$ckpt" "$train_tag" "$eval_tag" "$out_dir" "$run_tag" >> "$MANIFEST"
}

for ckpt in 500 1500 2500 3500 4500 5500 6500 7500 7764; do
  submit_one "$ckpt" with_metadata with_metadata "1b_sftckpt_screen_tplus_eplus_ckpt${ckpt}_seed41"
  submit_one "$ckpt" without_metadata without_metadata "1b_sftckpt_screen_tminus_eminus_ckpt${ckpt}_seed41"
done

echo "$MANIFEST"
