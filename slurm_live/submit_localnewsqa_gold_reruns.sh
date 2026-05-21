#!/usr/bin/env bash
set -euo pipefail

ROOT=/path/to/metacul
PYTHON=${PYTHON:-/path/to/nanotron-b200/bin/python}
LOG_DIR=/path/to/logs/slurm_logs
DATASET=${DATASET:-$ROOT/qa_data/localnewsqa_core/final_gold_20260516/localnewsqa_gold_explicit17000_ambiguous1700.jsonl}
SEEDS_CSV=${SEEDS:-41,42,43,44,45}
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT=${RUN_ROOT:-$ROOT/results/localnewsqa_gold_20260516}
MAPLE_WRAPPER=${MAPLE_WRAPPER:-$ROOT/slurm/pretrained_localnewsqa_eval_single_gpuq_sftfull.slurm}
BASELINE_WRAPPER=${BASELINE_WRAPPER:-$ROOT/slurm/localnewsqa_eval_baseline_single.slurm}
ADVERSARIAL_WRAPPER=${ADVERSARIAL_WRAPPER:-$ROOT/slurm/pretrained_localnewsqa_adversarial_grid.slurm}
SBATCH_RETRY_SLEEP=${SBATCH_RETRY_SLEEP:-300}

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
mkdir -p "$LOG_DIR" "$RUN_ROOT"

MANIFEST="$LOG_DIR/localnewsqa_gold_reruns_${STAMP}.tsv"
printf 'job_id\tgroup\tfamily\tseed\ttrain_meta\teval_meta\tlocale_role\tout_dir\trun_tag\textra\n' > "$MANIFEST"

PRETRAIN_TARGET_ROOT="$RUN_ROOT/pretrained_target"
PRETRAIN_CONTRAST_ROOT="$RUN_ROOT/pretrained_contrast"
SFT_TARGET_ROOT="$RUN_ROOT/sft_target"
SFT_CONTRAST_ROOT="$RUN_ROOT/sft_contrast"
EXTERNAL_TARGET_ROOT="$RUN_ROOT/external_target"
EXTERNAL_CONTRAST_ROOT="$RUN_ROOT/external_contrast"
ADVERSARIAL_ROOT="$RUN_ROOT/adversarial_full_mismatch"
URLMASK_ROOT="$RUN_ROOT/adversarial_urlmask"
SUMMARY_ROOT="$RUN_ROOT/summaries"
PLOT_ROOT="$RUN_ROOT/plots"

ALL_JOBS=()
PRETRAIN_JOBS=()
SFT_JOBS=()
EXTERNAL_JOBS=()
ADVERSARIAL_JOBS=()

PREVIOUS_MANIFESTS=${PREVIOUS_MANIFESTS:-}
declare -A SUBMITTED_TASKS=()

task_key() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' "$@"
}

remember_task() {
  local job_id="$1"
  local group="$2"
  local family="$3"
  local seed="$4"
  local train_meta="$5"
  local eval_meta="$6"
  local locale_role="$7"
  local out_dir="$8"
  local run_tag="$9"
  local extra="${10}"
  local key
  key=$(task_key "$group" "$family" "$seed" "$train_meta" "$eval_meta" "$locale_role" "$out_dir" "$run_tag" "$extra")
  SUBMITTED_TASKS["$key"]="$job_id"
  ALL_JOBS+=("$job_id")
  case "$group" in
    pretrained*) PRETRAIN_JOBS+=("$job_id") ;;
    sft*) SFT_JOBS+=("$job_id") ;;
    external*) EXTERNAL_JOBS+=("$job_id") ;;
    adversarial*) ADVERSARIAL_JOBS+=("$job_id") ;;
  esac
}

load_previous_manifests() {
  if [[ -z "$PREVIOUS_MANIFESTS" ]]; then
    return
  fi
  local manifest
  local previous_list="${PREVIOUS_MANIFESTS//,/ }"
  for manifest in $previous_list; do
    if [[ ! -f "$manifest" ]]; then
      printf '[warn] previous manifest not found: %s\n' "$manifest" >&2
      continue
    fi
    while IFS=$'\t' read -r job_id group family seed train_meta eval_meta locale_role out_dir run_tag extra; do
      if [[ "$job_id" == "job_id" || -z "$job_id" ]]; then
        continue
      fi
      remember_task "$job_id" "$group" "$family" "$seed" "$train_meta" "$eval_meta" "$locale_role" "$out_dir" "$run_tag" "$extra"
    done < "$manifest"
  done
  printf '[info] loaded %s previously submitted task keys\n' "${#SUBMITTED_TASKS[@]}" >&2
}

already_submitted() {
  local group="$1"
  local family="$2"
  local seed="$3"
  local train_meta="$4"
  local eval_meta="$5"
  local locale_role="$6"
  local out_dir="$7"
  local run_tag="$8"
  local extra="${9}"
  local key
  key=$(task_key "$group" "$family" "$seed" "$train_meta" "$eval_meta" "$locale_role" "$out_dir" "$run_tag" "$extra")
  if [[ -n "${SUBMITTED_TASKS[$key]:-}" ]]; then
    printf '[skip] already submitted job=%s group=%s family=%s seed=%s role=%s tag=%s\n' \
      "${SUBMITTED_TASKS[$key]}" "$group" "$family" "$seed" "$locale_role" "$run_tag" >&2
    return 0
  fi
  return 1
}

submit_sbatch() {
  local output status
  while true; do
    set +e
    output=$(sbatch --parsable "$@" 2>&1)
    status=$?
    set -e
    if [[ "$status" -eq 0 ]]; then
      printf '%s\n' "$output"
      return 0
    fi
    if grep -q "QOSMaxSubmitJobPerUserLimit" <<< "$output"; then
      printf '[wait] Slurm submit cap reached; retrying in %ss\n' "$SBATCH_RETRY_SLEEP" >&2
      sleep "$SBATCH_RETRY_SLEEP"
      continue
    fi
    printf '%s\n' "$output" >&2
    return "$status"
  done
}

submit_maple() {
  local group="$1"
  local family="$2"
  local seed="$3"
  local train_meta="$4"
  local eval_meta="$5"
  local locale_role="$6"
  local out_dir="$7"
  local run_tag="$8"
  local with_path="$9"
  local without_path="${10}"
  local metadata_prompt_style="${11}"
  local qa_prompt_style="${12}"
  local answer_cue_style="${13}"
  local omit_option_labels="${14}"
  local exact_option_text_instruction="${15}"
  local add_prompt_bos="${16}"
  local null_calibration_mode="${17}"
  local length_norm_alpha="${18}"
  local null_calibration_beta="${19}"
  local base_with="${20:-}"
  local base_without="${21:-}"
  local adapter_with="${22:-}"
  local adapter_without="${23:-}"

  if already_submitted "$group" "$family" "$seed" "$train_meta" "$eval_meta" "$locale_role" "$out_dir" "$run_tag" ""; then
    return
  fi

  mkdir -p "$out_dir"
  local job_id
  job_id=$(submit_sbatch \
    --export=ALL,MODEL_TYPE=custom,PYTHON="$PYTHON",DATASET="$DATASET",SPLIT=train,LOCALE_ROLE="$locale_role",META_TAG="$eval_meta",TRAIN_META_TAG="$train_meta",EVAL_META_TAG="$eval_meta",CUSTOM_MODEL_PATH_WITH="$with_path",CUSTOM_MODEL_PATH_WITHOUT="$without_path",BASE_MODEL_PATH_WITH="$base_with",BASE_MODEL_PATH_WITHOUT="$base_without",PEFT_ADAPTER_PATH_WITH="$adapter_with",PEFT_ADAPTER_PATH_WITHOUT="$adapter_without",OUT_DIR="$out_dir",RUN_TAG="$run_tag",SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED="$seed",OMIT_OPTION_LABELS="$omit_option_labels",EXACT_OPTION_TEXT_INSTRUCTION="$exact_option_text_instruction",MCQ_SCORING=option_text_avg,ANSWER_FORMAT=option,METADATA_PROMPT_STYLE="$metadata_prompt_style",QA_PROMPT_STYLE="$qa_prompt_style",ANSWER_CUE_STYLE="$answer_cue_style",SAMPLE_SEED="$seed",ADD_PROMPT_BOS="$add_prompt_bos",NULL_CALIBRATION_MODE="$null_calibration_mode",NULL_CALIBRATION_BETA="$null_calibration_beta",LENGTH_NORM_ALPHA="$length_norm_alpha",RESUME=1 \
    "$MAPLE_WRAPPER")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$job_id" "$group" "$family" "$seed" "$train_meta" "$eval_meta" "$locale_role" "$out_dir" "$run_tag" "" >> "$MANIFEST"
  ALL_JOBS+=("$job_id")
  case "$group" in
    pretrained*) PRETRAIN_JOBS+=("$job_id") ;;
    sft*) SFT_JOBS+=("$job_id") ;;
  esac
  local key
  key=$(task_key "$group" "$family" "$seed" "$train_meta" "$eval_meta" "$locale_role" "$out_dir" "$run_tag" "")
  SUBMITTED_TASKS["$key"]="$job_id"
}

submit_external() {
  local group="$1"
  local seed="$2"
  local model_slug="$3"
  local model_name="$4"
  local meta_tag="$5"
  local locale_role="$6"
  local out_dir="$7"
  local split_filter="$8"

  if already_submitted "$group" "$model_slug" "$seed" "" "$meta_tag" "$locale_role" "$out_dir" "" "$model_name"; then
    return
  fi

  mkdir -p "$out_dir"
  local job_id
  job_id=$(submit_sbatch \
    --export=ALL,DATASET="$DATASET",MODEL_NAME="$model_name",MODEL_SLUG="$model_slug",META_TAG="$meta_tag",LOCALE_ROLE="$locale_role",SPLIT_TYPE_FILTER="$split_filter",OUT_DIR="$out_dir",SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED="$seed",MCQ_SCORING=option_text_avg,ANSWER_FORMAT=option,RESUME=1 \
    "$BASELINE_WRAPPER")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$job_id" "$group" "$model_slug" "$seed" "" "$meta_tag" "$locale_role" "$out_dir" "" "$model_name" >> "$MANIFEST"
  ALL_JOBS+=("$job_id")
  EXTERNAL_JOBS+=("$job_id")
  local key
  key=$(task_key "$group" "$model_slug" "$seed" "" "$meta_tag" "$locale_role" "$out_dir" "" "$model_name")
  SUBMITTED_TASKS["$key"]="$job_id"
}

submit_adversarial() {
  local group="$1"
  local family="$2"
  local meta_tag="$3"
  local with_path="$4"
  local without_path="$5"
  local out_dir="$6"
  local run_tag="$7"
  local metadata_prompt_style="$8"
  local qa_prompt_style="$9"
  local answer_cue_style="${10}"
  local omit_option_labels="${11}"
  local exact_option_text_instruction="${12}"
  local add_prompt_bos="${13}"
  local null_calibration_mode="${14}"
  local length_norm_alpha="${15}"
  local null_calibration_beta="${16}"
  local rate_list="${17}"
  local corruption_mode="${18}"

  if already_submitted "$group" "$family" "" "" "$meta_tag" "target" "$out_dir" "$run_tag" "$corruption_mode"; then
    return
  fi

  mkdir -p "$out_dir"
  local job_id
  job_id=$(submit_sbatch \
    --export=ALL,DATASET="$DATASET",MODEL_TYPE=custom,META_TAG="$meta_tag",CUSTOM_MODEL_PATH_WITH="$with_path",CUSTOM_MODEL_PATH_WITHOUT="$without_path",OUT_DIR="$out_dir",RUN_TAG="$run_tag",RATE_LIST="$rate_list",URL_CORRUPTION_MODE="$corruption_mode",URL_MASK_TOKEN=xx,SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED=42,OMIT_OPTION_LABELS="$omit_option_labels",EXACT_OPTION_TEXT_INSTRUCTION="$exact_option_text_instruction",MCQ_SCORING=option_text_avg,METADATA_PROMPT_STYLE="$metadata_prompt_style",QA_PROMPT_STYLE="$qa_prompt_style",ANSWER_CUE_STYLE="$answer_cue_style",ADD_PROMPT_BOS="$add_prompt_bos",NULL_CALIBRATION_MODE="$null_calibration_mode",NULL_CALIBRATION_BETA="$null_calibration_beta",LENGTH_NORM_ALPHA="$length_norm_alpha" \
    "$ADVERSARIAL_WRAPPER")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$job_id" "$group" "$family" "" "" "$meta_tag" "target" "$out_dir" "$run_tag" "$corruption_mode" >> "$MANIFEST"
  ALL_JOBS+=("$job_id")
  ADVERSARIAL_JOBS+=("$job_id")
  local key
  key=$(task_key "$group" "$family" "" "" "$meta_tag" "target" "$out_dir" "$run_tag" "$corruption_mode")
  SUBMITTED_TASKS["$key"]="$job_id"
}

submit_summary() {
  local group="$1"
  local dep_jobs="$2"
  local command="$3"
  if [[ -z "$dep_jobs" ]]; then
    return
  fi
  local job_id
  job_id=$(submit_sbatch \
    --dependency=afterok:"$dep_jobs" \
    --job-name="$group" \
    --partition=normal \
    --mem=12G \
    --time=02:00:00 \
    --output="$LOG_DIR/${group}.%j.out" \
    --error="$LOG_DIR/${group}.%j.err" \
    --wrap="$command")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$job_id" "summary" "$group" "" "" "" "" "$SUMMARY_ROOT" "" "" >> "$MANIFEST"
  ALL_JOBS+=("$job_id")
}

echo "[info] dataset=$DATASET"
echo "[info] run_root=$RUN_ROOT"
load_previous_manifests

for seed in "${SEEDS[@]}"; do
  for combo in \
    "with_metadata with_metadata tplus eplus" \
    "with_metadata without_metadata tplus eminus" \
    "without_metadata with_metadata tminus eplus" \
    "without_metadata without_metadata tminus eminus"
  do
    read -r train_meta eval_meta train_short eval_short <<< "$combo"
    submit_maple pretrained_target 1b "$seed" "$train_meta" "$eval_meta" target \
      "$PRETRAIN_TARGET_ROOT/1b_codeg_labels_question_final/seed_${seed}" \
      "frozenfig9_1b_${train_short}_${eval_short}_seed${seed}" \
      "$ROOT/models/combined_with_metadata_1b" "$ROOT/models/combined_without_metadata_1b" \
      code_grounded question final_answer_colon 0 1 0 none 1.25 0.0
    submit_maple pretrained_contrast 1b "$seed" "$train_meta" "$eval_meta" contrast \
      "$PRETRAIN_CONTRAST_ROOT/1b_codeg_labels_question_final/seed_${seed}" \
      "frozenfig9contrast_1b_${train_short}_${eval_short}_seed${seed}" \
      "$ROOT/models/combined_with_metadata_1b" "$ROOT/models/combined_without_metadata_1b" \
      code_grounded question final_answer_colon 0 1 0 none 1.25 0.0

    submit_maple pretrained_target 3b "$seed" "$train_meta" "$eval_meta" target \
      "$PRETRAIN_TARGET_ROOT/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_${seed}" \
      "frozenfig9_3b_${train_short}_${eval_short}_seed${seed}" \
      "$ROOT/models/combined_with_metadata_3b" "$ROOT/models/combined_without_metadata_3b" \
      name_grounded question_answer final_answer_colon 1 0 1 question_masked 0.25 0.5
    submit_maple pretrained_contrast 3b "$seed" "$train_meta" "$eval_meta" contrast \
      "$PRETRAIN_CONTRAST_ROOT/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_${seed}" \
      "frozenfig9contrast_3b_${train_short}_${eval_short}_seed${seed}" \
      "$ROOT/models/combined_with_metadata_3b" "$ROOT/models/combined_without_metadata_3b" \
      name_grounded question_answer final_answer_colon 1 0 1 question_masked 0.25 0.5

    submit_maple sft_target 1b "$seed" "$train_meta" "$eval_meta" target \
      "$SFT_TARGET_ROOT/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_${seed}" \
      "sftgold_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_${train_short}_${eval_short}_seed${seed}" \
      "$ROOT/models/sft/combined_with_metadata_chat" "$ROOT/models/sft/combined_without_metadata_chat" \
      name_plain question_answer country_final_answer_colon 1 0 1 question_masked 0.25 0.25
    submit_maple sft_contrast 1b "$seed" "$train_meta" "$eval_meta" contrast \
      "$SFT_CONTRAST_ROOT/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_${seed}" \
      "sftgoldcontrast_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_${train_short}_${eval_short}_seed${seed}" \
      "$ROOT/models/sft/combined_with_metadata_chat" "$ROOT/models/sft/combined_without_metadata_chat" \
      name_plain question_answer country_final_answer_colon 1 0 1 question_masked 0.25 0.25

    submit_maple sft_target 3b "$seed" "$train_meta" "$eval_meta" target \
      "$SFT_TARGET_ROOT/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_${seed}" \
      "sftgold_3b_best3b_chat_${train_short}_${eval_short}_seed${seed}" \
      "$ROOT/models/sft/combined_with_metadata_3b_best3b_chat" "$ROOT/models/sft/combined_without_metadata_3b_best3b_chat" \
      name_grounded question_answer final_answer_colon 1 0 1 question_masked 0.25 0.5
    submit_maple sft_contrast 3b "$seed" "$train_meta" "$eval_meta" contrast \
      "$SFT_CONTRAST_ROOT/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_${seed}" \
      "sftgoldcontrast_3b_best3b_chat_${train_short}_${eval_short}_seed${seed}" \
      "$ROOT/models/sft/combined_with_metadata_3b_best3b_chat" "$ROOT/models/sft/combined_without_metadata_3b_best3b_chat" \
      name_grounded question_answer final_answer_colon 1 0 1 question_masked 0.25 0.5
  done
done

EXTERNAL_MODELS=(
  "llama32_1b|meta-llama/Llama-3.2-1B-Instruct"
  "llama32_3b|meta-llama/Llama-3.2-3B-Instruct"
  "qwen25_0p5b|Qwen/Qwen2.5-0.5B-Instruct"
  "qwen25_1p5b|Qwen/Qwen2.5-1.5B-Instruct"
  "qwen25_3b|Qwen/Qwen2.5-3B-Instruct"
  "qwen35_0p8b|Qwen/Qwen3.5-0.8B"
  "qwen35_2b|Qwen/Qwen3.5-2B"
  "qwen35_4b|Qwen/Qwen3.5-4B"
  "qwen35_9b|Qwen/Qwen3.5-9B"
  "gemma4_e2b_it|google/gemma-4-E2B-it"
  "gemma4_e4b_it|google/gemma-4-E4B-it"
  "mistral7b_v02|mistralai/Mistral-7B-Instruct-v0.2"
)

for seed in "${SEEDS[@]}"; do
  for entry in "${EXTERNAL_MODELS[@]}"; do
    IFS='|' read -r model_slug model_name <<< "$entry"
    for meta_tag in with_metadata without_metadata; do
      submit_external external_target "$seed" "$model_slug" "$model_name" "$meta_tag" target \
        "$EXTERNAL_TARGET_ROOT/seed_${seed}" all
      submit_external external_contrast "$seed" "$model_slug" "$model_name" "$meta_tag" contrast \
        "$EXTERNAL_CONTRAST_ROOT/seed_${seed}" ambiguous
    done
  done
done

submit_adversarial adversarial_full_mismatch 1b with_metadata \
  "$ROOT/models/combined_with_metadata_1b" "$ROOT/models/combined_without_metadata_1b" \
  "$ADVERSARIAL_ROOT/1b" 1b_tplus_adv \
  code_grounded question final_answer_colon 0 1 0 none 1.25 0.0 '0;0.25;0.5;0.75;1.0' full_mismatch
submit_adversarial adversarial_full_mismatch 1b without_metadata \
  "$ROOT/models/combined_with_metadata_1b" "$ROOT/models/combined_without_metadata_1b" \
  "$ADVERSARIAL_ROOT/1b" 1b_tminus_adv \
  code_grounded question final_answer_colon 0 1 0 none 1.25 0.0 '0' full_mismatch
submit_adversarial adversarial_full_mismatch 3b with_metadata \
  "$ROOT/models/combined_with_metadata_3b" "$ROOT/models/combined_without_metadata_3b" \
  "$ADVERSARIAL_ROOT/3b" 3b_tplus_adv \
  name_grounded question_answer final_answer_colon 1 0 1 question_masked 0.25 0.5 '0;0.25;0.5;0.75;1.0' full_mismatch
submit_adversarial adversarial_full_mismatch 3b without_metadata \
  "$ROOT/models/combined_with_metadata_3b" "$ROOT/models/combined_without_metadata_3b" \
  "$ADVERSARIAL_ROOT/3b" 3b_tminus_adv \
  name_grounded question_answer final_answer_colon 1 0 1 question_masked 0.25 0.5 '0' full_mismatch

submit_adversarial adversarial_urlmask 1b with_metadata \
  "$ROOT/models/combined_with_metadata_1b" "$ROOT/models/combined_without_metadata_1b" \
  "$URLMASK_ROOT/1b" 1b_tplus_urlmask \
  code_grounded question final_answer_colon 0 1 0 none 1.25 0.0 '0;0.25;0.5;0.75;1.0' url_country_mask
submit_adversarial adversarial_urlmask 1b without_metadata \
  "$ROOT/models/combined_with_metadata_1b" "$ROOT/models/combined_without_metadata_1b" \
  "$URLMASK_ROOT/1b" 1b_tminus_urlmask \
  code_grounded question final_answer_colon 0 1 0 none 1.25 0.0 '0' url_country_mask
submit_adversarial adversarial_urlmask 3b with_metadata \
  "$ROOT/models/combined_with_metadata_3b" "$ROOT/models/combined_without_metadata_3b" \
  "$URLMASK_ROOT/3b" 3b_tplus_urlmask \
  name_grounded question_answer final_answer_colon 1 0 1 question_masked 0.25 0.5 '0;0.25;0.5;0.75;1.0' url_country_mask
submit_adversarial adversarial_urlmask 3b without_metadata \
  "$ROOT/models/combined_with_metadata_3b" "$ROOT/models/combined_without_metadata_3b" \
  "$URLMASK_ROOT/3b" 3b_tminus_urlmask \
  name_grounded question_answer final_answer_colon 1 0 1 question_masked 0.25 0.5 '0' url_country_mask

mkdir -p "$SUMMARY_ROOT" "$PLOT_ROOT"

if ((${#PRETRAIN_JOBS[@]} > 0)); then
  PRETRAIN_DEP=$(IFS=:; echo "${PRETRAIN_JOBS[*]}")
  submit_summary lnqa-gold-pre-sum "$PRETRAIN_DEP" \
    "$PYTHON $ROOT/src/50_localnewsqa_pretrained_frozen_multiseed_summary.py --root $PRETRAIN_TARGET_ROOT --plot-csv $PLOT_ROOT/plot_8_pretrained_target_split_multiseed.csv && $PYTHON $ROOT/src/68_localnewsqa_seed41_bootstrap_summary.py --input-root $PRETRAIN_TARGET_ROOT --output-csv $PLOT_ROOT/plot_8_pretrained_target_split_seed41_bootstrap.csv && $PYTHON $ROOT/src/55_localnewsqa_locale_switch.py --target-root-1b $PRETRAIN_TARGET_ROOT/1b_codeg_labels_question_final --contrast-root-1b $PRETRAIN_CONTRAST_ROOT/1b_codeg_labels_question_final --target-root-3b $PRETRAIN_TARGET_ROOT/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos --contrast-root-3b $PRETRAIN_CONTRAST_ROOT/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos --output-root $SUMMARY_ROOT/localnewsqa_locale_switch"
fi

if ((${#SFT_JOBS[@]} > 0)); then
  SFT_DEP=$(IFS=:; echo "${SFT_JOBS[*]}")
  submit_summary lnqa-gold-sft-sum "$SFT_DEP" \
    "$PYTHON $ROOT/src/71_localnewsqa_gold_sft_summary.py --sft-root $SFT_TARGET_ROOT --plot-csv $PLOT_ROOT/plot_8_sft_target_split_multiseed_gold.csv && $PYTHON $ROOT/src/61_localnewsqa_sft_locale_switch.py --target-root-1b $SFT_TARGET_ROOT/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos --contrast-root-1b $SFT_CONTRAST_ROOT/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos --target-root-3b $SFT_TARGET_ROOT/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos --contrast-root-3b $SFT_CONTRAST_ROOT/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos --output-root $SUMMARY_ROOT/localnewsqa_sft_locale_switch"
fi

if ((${#EXTERNAL_JOBS[@]} > 0)); then
  EXTERNAL_DEP=$(IFS=:; echo "${EXTERNAL_JOBS[*]}")
  submit_summary lnqa-gold-ext-sum "$EXTERNAL_DEP" \
    "$PYTHON $ROOT/src/35_localnewsqa_multiseed_summary.py --pretrained-root $PRETRAIN_TARGET_ROOT/not_used --baseline-root $EXTERNAL_TARGET_ROOT --plot-csv $PLOT_ROOT/unused_pretrained_from_external_summary.csv --table7-long-csv $PLOT_ROOT/table7_external_localnewsqa_multiseed_long.csv --table7-wide-csv $PLOT_ROOT/table7_external_localnewsqa_multiseed_wide.csv && $PYTHON $ROOT/src/56_localnewsqa_external_locale_switch.py --target-root $EXTERNAL_TARGET_ROOT --contrast-root $EXTERNAL_CONTRAST_ROOT --output-root $SUMMARY_ROOT/localnewsqa_external_locale_switch"
fi

if ((${#ADVERSARIAL_JOBS[@]} > 0)); then
  ADV_DEP=$(IFS=:; echo "${ADVERSARIAL_JOBS[*]}")
  submit_summary lnqa-gold-adv-sum "$ADV_DEP" \
    "$PYTHON $ROOT/src/44_pretrained_adversarial_summary.py --input-dir $ADVERSARIAL_ROOT --output-csv $PLOT_ROOT/adversarial_pretrained_summary_full_mismatch.csv && $PYTHON $ROOT/src/44_pretrained_adversarial_summary.py --input-dir $URLMASK_ROOT --output-csv $PLOT_ROOT/adversarial_pretrained_summary_urlmask.csv"
fi

printf '[done] manifest=%s\n' "$MANIFEST"
printf '[done] total_jobs_including_summaries=%s\n' "${#ALL_JOBS[@]}"
