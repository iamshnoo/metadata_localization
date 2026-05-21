#!/bin/bash
set -euo pipefail

CKPT="${1:?usage: $0 <checkpoint-step>}"

SLURM=${SLURM_SCRIPT:-/path/to/metacul/slurm/pretrained_localnewsqa_eval_single.slurm}
LOG_DIR=/path/to/logs/slurm_logs
SUMMARY_SH=/path/to/metacul/slurm/submit_figure9_sft_1b_altprompt_summary.sh
RESULT_ROOT=${RESULT_ROOT:-/path/to/metacul/results/downstream_localnewsqa_sft_figure9_1b_checkpoint_multiseed/ckpt_${CKPT}}
RUN_ROOT_NAME=${RUN_ROOT_NAME:-1b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos}
RUN_TAG_PREFIX=${RUN_TAG_PREFIX:-1b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos}
ONE_B_PLOT_CSV=${ONE_B_PLOT_CSV:-/path/to/metacul/results/plots/plot8/plot_8_sft_target_split_multiseed_1b_checkpoint.csv}
REQUIREMENT_ONLY=${REQUIREMENT_ONLY:-0}
OMIT_OPTION_LABELS=${OMIT_OPTION_LABELS:-1}
EXACT_OPTION_TEXT_INSTRUCTION=${EXACT_OPTION_TEXT_INSTRUCTION:-0}
MCQ_SCORING=${MCQ_SCORING:-option_text_avg}
METADATA_PROMPT_STYLE=${METADATA_PROMPT_STYLE:-name_grounded}
QA_PROMPT_STYLE=${QA_PROMPT_STYLE:-question_answer}
ANSWER_CUE_STYLE=${ANSWER_CUE_STYLE:-final_answer_colon}
ADD_PROMPT_BOS=${ADD_PROMPT_BOS:-1}
NULL_CALIBRATION_MODE=${NULL_CALIBRATION_MODE:-question_masked}
NULL_CALIBRATION_BETA=${NULL_CALIBRATION_BETA:-0.5}
LENGTH_NORM_ALPHA=${LENGTH_NORM_ALPHA:-0.25}
RESUME=${RESUME:-0}
TS=$(date +%Y%m%d_%H%M%S)
MANIFEST=$LOG_DIR/figure9_sft_1b_checkpoint_multiseed_ckpt${CKPT}_${TS}.tsv

WITH_BASE=/path/to/metacul/models/combined_with_metadata_1b
WITHOUT_BASE=/path/to/metacul/models/combined_without_metadata_1b
WITH_ADAPTER=/path/to/metacul/models/sft/combined_with_metadata_sft_lora/checkpoint-${CKPT}
WITHOUT_ADAPTER=/path/to/metacul/models/sft/combined_without_metadata_sft_lora/checkpoint-${CKPT}

mkdir -p "$LOG_DIR" "$RESULT_ROOT"
printf 'jobid\tcheckpoint\tseed\ttrain_meta\teval_meta\tout_dir\trun_tag\n' > "$MANIFEST"

submit_family() {
  local seed="$1"
  local out_dir="$RESULT_ROOT/${RUN_ROOT_NAME}/seed_${seed}"
  mkdir -p "$out_dir"

  local combos=()
  if [[ "$REQUIREMENT_ONLY" == "1" ]]; then
    combos=(
      "with_metadata with_metadata tplus eplus"
      "without_metadata without_metadata tminus eminus"
    )
  else
    combos=(
      "with_metadata with_metadata tplus eplus"
      "with_metadata without_metadata tplus eminus"
      "without_metadata with_metadata tminus eplus"
      "without_metadata without_metadata tminus eminus"
    )
  fi

  for combo in "${combos[@]}"; do
    read -r train_tag eval_tag train_short eval_short <<< "$combo"
    run_tag="${RUN_TAG_PREFIX}_${train_short}_${eval_short}_seed${seed}"
    jobid=$(sbatch --parsable \
      --export=ALL,MODEL_TYPE=custom,META_TAG="$eval_tag",TRAIN_META_TAG="$train_tag",EVAL_META_TAG="$eval_tag",BASE_MODEL_PATH_WITH="$WITH_BASE",BASE_MODEL_PATH_WITHOUT="$WITHOUT_BASE",PEFT_ADAPTER_PATH_WITH="$WITH_ADAPTER",PEFT_ADAPTER_PATH_WITHOUT="$WITHOUT_ADAPTER",OUT_DIR="$out_dir",RUN_TAG="$run_tag",SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED="$seed",OMIT_OPTION_LABELS="$OMIT_OPTION_LABELS",EXACT_OPTION_TEXT_INSTRUCTION="$EXACT_OPTION_TEXT_INSTRUCTION",MCQ_SCORING="$MCQ_SCORING",METADATA_PROMPT_STYLE="$METADATA_PROMPT_STYLE",QA_PROMPT_STYLE="$QA_PROMPT_STYLE",ANSWER_CUE_STYLE="$ANSWER_CUE_STYLE",SAMPLE_SEED="$seed",ADD_PROMPT_BOS="$ADD_PROMPT_BOS",NULL_CALIBRATION_MODE="$NULL_CALIBRATION_MODE",NULL_CALIBRATION_BETA="$NULL_CALIBRATION_BETA",LENGTH_NORM_ALPHA="$LENGTH_NORM_ALPHA",RESUME="$RESUME" \
      "$SLURM")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$jobid" "$CKPT" "$seed" "$train_tag" "$eval_tag" "$out_dir" "$run_tag" >> "$MANIFEST"
  done
}

for seed in 41 42 43 44 45; do
  submit_family "$seed"
done

SUMMARY_JOB=$(bash "$SUMMARY_SH" "$MANIFEST" "$RESULT_ROOT" "$ONE_B_PLOT_CSV")
echo "summary_jobid=${SUMMARY_JOB}"
echo "$MANIFEST"
