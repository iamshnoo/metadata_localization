#!/bin/bash
set -euo pipefail

ROOT=/path/to/metacul
LOG_DIR=/path/to/logs/slurm_logs
mkdir -p "$LOG_DIR"

SEEDS_CSV=${SEEDS:-41,42,43,44,45}
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"

TARGET_ROOT=${TARGET_ROOT:-$ROOT/results/downstream_localnewsqa_external_baselines_multiseed}
CONTRAST_ROOT=${CONTRAST_ROOT:-$ROOT/results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun}
SUMMARY_ROOT=${SUMMARY_ROOT:-$ROOT/results/analysis/localnewsqa_external_locale_switch_rerun}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="$LOG_DIR/external_localnewsqa_contrast_multiseed_rerun_${TIMESTAMP}.tsv"
printf "job_id\tgroup\tseed\tlabel\tscript\n" > "$MANIFEST"

JOB_IDS=()

submit_job() {
  local group="$1"
  local seed="$2"
  local label="$3"
  local script="$4"
  local export_vars="$5"
  local job_id
  job_id=$(sbatch --parsable --export="$export_vars" "$script")
  JOB_IDS+=("$job_id")
  printf "%s\t%s\t%s\t%s\t%s\n" "$job_id" "$group" "$seed" "$label" "$script" >> "$MANIFEST"
  printf "[submitted] %s seed=%s label=%s job=%s\n" "$group" "$seed" "$label" "$job_id"
}

TABLE7_MODELS=(
  "llama32_1b|meta-llama/Llama-3.2-1B-Instruct"
  "llama32_3b|meta-llama/Llama-3.2-3B-Instruct"
  "qwen25_1p5b|Qwen/Qwen2.5-1.5B-Instruct"
  "qwen25_3b|Qwen/Qwen2.5-3B-Instruct"
  "qwen35_0p8b|Qwen/Qwen3.5-0.8B"
  "qwen35_2b|Qwen/Qwen3.5-2B"
  "qwen35_4b|Qwen/Qwen3.5-4B"
  "gemma4_e2b_it|google/gemma-4-E2B-it"
  "gemma4_e4b_it|google/gemma-4-E4B-it"
)

for seed in "${SEEDS[@]}"; do
  contrast_out_dir="$CONTRAST_ROOT/seed_${seed}"
  for entry in "${TABLE7_MODELS[@]}"; do
    IFS='|' read -r model_slug model_name <<< "$entry"
    for meta_tag in with_metadata without_metadata; do
      submit_job \
        "table7_pair" \
        "$seed" \
        "${model_slug}_${meta_tag}_contrast" \
        "$ROOT/slurm/localnewsqa_eval_baseline_single.slurm" \
        "ALL,MODEL_NAME=${model_name},MODEL_SLUG=${model_slug},META_TAG=${meta_tag},LOCALE_ROLE=contrast,SPLIT_TYPE_FILTER=ambiguous,OUT_DIR=${contrast_out_dir},SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED=${seed},MCQ_SCORING=option_text_avg,ANSWER_FORMAT=option"
    done
  done
done

if ((${#JOB_IDS[@]} > 0)); then
  DEP=$(IFS=:; echo "${JOB_IDS[*]}")
  sbatch --parsable \
    --dependency=afterok:${DEP} \
    --job-name=ext-lnqa-pair-sum \
    --output="$LOG_DIR/ext_lnqa_pair_sum_rerun.%j.out" \
    --error="$LOG_DIR/ext_lnqa_pair_sum_rerun.%j.err" \
    --wrap="python3 $ROOT/src/56_localnewsqa_external_locale_switch.py --target-root $TARGET_ROOT --contrast-root $CONTRAST_ROOT --output-root $SUMMARY_ROOT"
fi

printf "[done] Manifest written to %s\n" "$MANIFEST"
