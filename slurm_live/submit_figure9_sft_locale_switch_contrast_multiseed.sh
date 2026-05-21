#!/usr/bin/env bash
set -euo pipefail

ROOT=/path/to/metacul
LOG_DIR=/path/to/logs/slurm_logs
mkdir -p "$LOG_DIR"

WRAPPER="$ROOT/slurm/pretrained_localnewsqa_eval_single_gpuq_sftfull.slurm"
PYTHON=/path/to/nanotron-b200/bin/python
SUMMARY_SCRIPT="$ROOT/src/61_localnewsqa_sft_locale_switch.py"
STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="$LOG_DIR/figure9_sft_locale_switch_contrast_multiseed_${STAMP}.tsv"

echo -e "jobid\tfamily\tseed\ttrain_meta\teval_meta\tout_dir\trun_tag" > "$MANIFEST"

submit_one() {
  local family="$1"
  local seed="$2"
  local train_meta="$3"
  local eval_meta="$4"
  local out_dir="$5"
  local run_tag="$6"
  local metadata_prompt_style="$7"
  local qa_prompt_style="$8"
  local answer_cue_style="$9"
  local null_calibration_mode="${10}"
  local null_calibration_beta="${11}"
  local length_norm_alpha="${12}"
  local add_prompt_bos="${13}"
  local custom_model_path_with="${14}"
  local custom_model_path_without="${15}"
  local base_model_path_with="${16}"
  local base_model_path_without="${17}"
  local peft_adapter_path_with="${18}"
  local peft_adapter_path_without="${19}"

  local jobid
  jobid=$(sbatch --parsable \
    --export=ALL,\
MODEL_TYPE=custom,\
PYTHON="$PYTHON",\
LOCALE_ROLE=contrast,\
OUT_DIR="$out_dir",\
RUN_TAG="$run_tag",\
META_TAG="$eval_meta",\
TRAIN_META_TAG="$train_meta",\
EVAL_META_TAG="$eval_meta",\
SHUFFLE_OPTIONS=1,\
OPTION_SHUFFLE_SEED="$seed",\
OMIT_OPTION_LABELS=1,\
MCQ_SCORING=option_text_avg,\
ANSWER_FORMAT=option,\
METADATA_PROMPT_STYLE="$metadata_prompt_style",\
QA_PROMPT_STYLE="$qa_prompt_style",\
ANSWER_CUE_STYLE="$answer_cue_style",\
NULL_CALIBRATION_MODE="$null_calibration_mode",\
NULL_CALIBRATION_BETA="$null_calibration_beta",\
LENGTH_NORM_ALPHA="$length_norm_alpha",\
ADD_PROMPT_BOS="$add_prompt_bos",\
CUSTOM_MODEL_PATH_WITH="$custom_model_path_with",\
CUSTOM_MODEL_PATH_WITHOUT="$custom_model_path_without",\
BASE_MODEL_PATH_WITH="$base_model_path_with",\
BASE_MODEL_PATH_WITHOUT="$base_model_path_without",\
PEFT_ADAPTER_PATH_WITH="$peft_adapter_path_with",\
PEFT_ADAPTER_PATH_WITHOUT="$peft_adapter_path_without" \
    "$WRAPPER")
  echo -e "${jobid}\t${family}\t${seed}\t${train_meta}\t${eval_meta}\t${out_dir}\t${run_tag}" >> "$MANIFEST"
  JOB_IDS+=("$jobid")
}

JOB_IDS=()

for seed in 41 42 43 44 45; do
  one_b_root="$ROOT/results/downstream_localnewsqa_sft_figure9_1b_checkpoint_nameplain_countrycue_beta025_contrast_multiseed/ckpt_7764/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_${seed}"
  submit_one "1B_chat" "$seed" "with_metadata" "with_metadata" "$one_b_root" \
    "1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tplus_eplus_seed${seed}" \
    "name_plain" "question_answer" "country_final_answer_colon" "question_masked" "0.25" "0.25" "1" \
    "" "" \
    "$ROOT/models/combined_with_metadata_1b" "$ROOT/models/combined_without_metadata_1b" \
    "$ROOT/models/sft/combined_with_metadata_sft_lora/checkpoint-7764" "$ROOT/models/sft/combined_without_metadata_sft_lora/checkpoint-7764"

  submit_one "1B_chat" "$seed" "without_metadata" "without_metadata" "$one_b_root" \
    "1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tminus_eminus_seed${seed}" \
    "name_plain" "question_answer" "country_final_answer_colon" "question_masked" "0.25" "0.25" "1" \
    "" "" \
    "$ROOT/models/combined_with_metadata_1b" "$ROOT/models/combined_without_metadata_1b" \
    "$ROOT/models/sft/combined_with_metadata_sft_lora/checkpoint-7764" "$ROOT/models/sft/combined_without_metadata_sft_lora/checkpoint-7764"

  three_b_root="$ROOT/results/downstream_localnewsqa_sft_figure9_contrast_full_multiseed/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_${seed}"
  submit_one "3B_chat" "$seed" "with_metadata" "with_metadata" "$three_b_root" \
    "sftfig9_3b_best3b_chat_tplus_eplus_seed${seed}" \
    "name_grounded" "question_answer" "final_answer_colon" "question_masked" "0.5" "0.25" "1" \
    "$ROOT/models/sft/combined_with_metadata_3b_best3b_chat" "$ROOT/models/sft/combined_without_metadata_3b_best3b_chat" \
    "" "" "" ""

  submit_one "3B_chat" "$seed" "without_metadata" "without_metadata" "$three_b_root" \
    "sftfig9_3b_best3b_chat_tminus_eminus_seed${seed}" \
    "name_grounded" "question_answer" "final_answer_colon" "question_masked" "0.5" "0.25" "1" \
    "$ROOT/models/sft/combined_with_metadata_3b_best3b_chat" "$ROOT/models/sft/combined_without_metadata_3b_best3b_chat" \
    "" "" "" ""
done

deps=$(IFS=:; echo "${JOB_IDS[*]}")
summary_job=$(sbatch --parsable --dependency="afterok:${deps}" --job-name=sftpairsum --partition=normal --mem=8G --time=01:00:00 \
  --output="$LOG_DIR/sftpairsum.%j.out" --error="$LOG_DIR/sftpairsum.%j.err" \
  --wrap="$PYTHON $SUMMARY_SCRIPT")

echo "manifest: $MANIFEST"
echo "summary_job: $summary_job"
