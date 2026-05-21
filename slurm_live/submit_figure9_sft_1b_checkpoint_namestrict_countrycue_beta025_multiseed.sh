#!/bin/bash
set -euo pipefail

CKPT="${1:?usage: $0 <checkpoint-step>}"

SLURM=/path/to/metacul/slurm/pretrained_localnewsqa_eval_single.slurm
LOG_DIR=/path/to/logs/slurm_logs
SUMMARY_SH=/path/to/metacul/slurm/submit_figure9_sft_1b_altprompt_summary.sh
RESULT_ROOT=/path/to/metacul/results/downstream_localnewsqa_sft_figure9_1b_checkpoint_namestrict_countrycue_beta025_multiseed/ckpt_${CKPT}
PLOT_CSV=/path/to/metacul/results/plots/plot8/plot_8_sft_target_split_multiseed_1b_checkpoint_namestrict_countrycue_beta025.csv
TS=$(date +%Y%m%d_%H%M%S)
MANIFEST=$LOG_DIR/figure9_sft_1b_checkpoint_namestrict_countrycue_beta025_multiseed_ckpt${CKPT}_${TS}.tsv

WITH_BASE=/path/to/metacul/models/combined_with_metadata_1b
WITHOUT_BASE=/path/to/metacul/models/combined_without_metadata_1b
WITH_ADAPTER=/path/to/metacul/models/sft/combined_with_metadata_sft_lora/checkpoint-${CKPT}
WITHOUT_ADAPTER=/path/to/metacul/models/sft/combined_without_metadata_sft_lora/checkpoint-${CKPT}

mkdir -p "$LOG_DIR" "$RESULT_ROOT"
printf 'jobid\tcheckpoint\tseed\ttrain_meta\teval_meta\tout_dir\trun_tag\n' > "$MANIFEST"

submit_family() {
  local seed="$1"
  local out_dir="$RESULT_ROOT/1b_chat_name_strict_country_final_qanswer_nolabel_qmask025_bos/seed_${seed}"
  mkdir -p "$out_dir"

  for combo in \
    "with_metadata with_metadata tplus eplus" \
    "without_metadata without_metadata tminus eminus"
  do
    read -r train_tag eval_tag train_short eval_short <<< "$combo"
    run_tag="1b_chat_name_strict_country_final_qanswer_nolabel_qmask025_bos_${train_short}_${eval_short}_seed${seed}"
    jobid=$(sbatch --parsable \
      --export=ALL,MODEL_TYPE=custom,META_TAG="$eval_tag",TRAIN_META_TAG="$train_tag",EVAL_META_TAG="$eval_tag",BASE_MODEL_PATH_WITH="$WITH_BASE",BASE_MODEL_PATH_WITHOUT="$WITHOUT_BASE",PEFT_ADAPTER_PATH_WITH="$WITH_ADAPTER",PEFT_ADAPTER_PATH_WITHOUT="$WITHOUT_ADAPTER",OUT_DIR="$out_dir",RUN_TAG="$run_tag",SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED="$seed",OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,METADATA_PROMPT_STYLE=name_strict,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,SAMPLE_SEED="$seed",ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.25,LENGTH_NORM_ALPHA=0.25 \
      "$SLURM")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$jobid" "$CKPT" "$seed" "$train_tag" "$eval_tag" "$out_dir" "$run_tag" >> "$MANIFEST"
  done
}

for seed in 41 42 43 44 45; do
  submit_family "$seed"
done

echo "$MANIFEST"
bash "$SUMMARY_SH" "$MANIFEST" "$RESULT_ROOT" "$PLOT_CSV" >/dev/null
