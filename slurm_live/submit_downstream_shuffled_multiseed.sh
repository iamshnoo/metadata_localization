#!/bin/bash
set -euo pipefail

ROOT=/scratch/amukher6/metacul
LOG_DIR=/scratch/amukher6/logs/slurm_logs
mkdir -p "$LOG_DIR"

SEEDS_CSV=${SEEDS:-41,42,43,44,45}
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="$LOG_DIR/downstream_shuffled_multiseed_${TIMESTAMP}.tsv"
printf "job_id\tgroup\tseed\tlabel\tscript\n" > "$MANIFEST"

submit_job() {
  local group="$1"
  local seed="$2"
  local label="$3"
  local script="$4"
  local export_vars="$5"
  local job_id
  job_id=$(sbatch --parsable --export="$export_vars" "$script")
  printf "%s\t%s\t%s\t%s\t%s\n" "$job_id" "$group" "$seed" "$label" "$script" >> "$MANIFEST"
  printf "[submitted] %s seed=%s label=%s job=%s\n" "$group" "$seed" "$label" "$job_id"
}

for seed in "${SEEDS[@]}"; do
  pretrained_out_dir="$ROOT/results/downstream_localnewsqa_pretrained_multiseed/seed_${seed}"
  for size in 1b 3b; do
    with_path="$ROOT/models/combined_with_metadata_${size}"
    without_path="$ROOT/models/combined_without_metadata_${size}"
    for train_meta in with_metadata without_metadata; do
      for eval_meta in with_metadata without_metadata; do
        if [[ "$train_meta" == "with_metadata" ]]; then
          train_short=tplus
        else
          train_short=tminus
        fi
        if [[ "$eval_meta" == "with_metadata" ]]; then
          eval_short=eplus
        else
          eval_short=eminus
        fi
        run_tag="${size}_${train_short}_${eval_short}"
        submit_job \
          "figure9" \
          "$seed" \
          "$run_tag" \
          "$ROOT/slurm/pretrained_localnewsqa_eval_single.slurm" \
          "ALL,MODEL_TYPE=custom,META_TAG=${eval_meta},TRAIN_META_TAG=${train_meta},EVAL_META_TAG=${eval_meta},CUSTOM_MODEL_PATH_WITH=${with_path},CUSTOM_MODEL_PATH_WITHOUT=${without_path},OUT_DIR=${pretrained_out_dir},RUN_TAG=${run_tag},LOCALE_ROLE=target,SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED=${seed},MCQ_SCORING=option_text_avg,ANSWER_FORMAT=option"
      done
    done
  done
done

TABLE7_MODELS=(
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
  baseline_out_dir="$ROOT/results/downstream_localnewsqa_external_baselines_multiseed/seed_${seed}"
  for entry in "${TABLE7_MODELS[@]}"; do
    IFS='|' read -r model_slug model_name <<< "$entry"
    for meta_tag in with_metadata without_metadata; do
      submit_job \
        "table7" \
        "$seed" \
        "${model_slug}_${meta_tag}" \
        "$ROOT/slurm/localnewsqa_eval_baseline_single.slurm" \
        "ALL,MODEL_NAME=${model_name},MODEL_SLUG=${model_slug},META_TAG=${meta_tag},LOCALE_ROLE=target,OUT_DIR=${baseline_out_dir},SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED=${seed},MCQ_SCORING=option_text_avg,ANSWER_FORMAT=option"
    done
  done
done

TABLE8_VARIANTS_OURS=(
  "custom_tplus_eplus"
  "custom_tplus_eminus"
  "custom_tminus_eplus"
  "custom_tminus_eminus"
)
TABLE8_VARIANTS_LLAMA=(
  "llama3_chat_with_metadata"
  "llama3_chat_without_metadata"
)
TABLE8_BENCHMARKS=(geomlama globalopinionqa worldvaluebench mmlu)

for seed in "${SEEDS[@]}"; do
  for size in 1b 3b; do
    out_root="$ROOT/results/external_benchmarks_pretrained_multiseed/ours_${size}"
    with_path="$ROOT/models/combined_with_metadata_${size}"
    without_path="$ROOT/models/combined_without_metadata_${size}"
    for benchmark in "${TABLE8_BENCHMARKS[@]}"; do
      for variant in "${TABLE8_VARIANTS_OURS[@]}"; do
        submit_job \
          "table8" \
          "$seed" \
          "ours_${size}_${benchmark}_${variant}" \
          "$ROOT/slurm/pretrained_external_eval_single.slurm" \
          "ALL,BENCHMARK=${benchmark},VARIANT=${variant},OUT_ROOT=${out_root},CUSTOM_MODEL_PATH_WITH=${with_path},CUSTOM_MODEL_PATH_WITHOUT=${without_path},EVAL_SEED=${seed},SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED=${seed}"
      done
    done
  done

  llama_out_root="$ROOT/results/external_benchmarks_pretrained_multiseed/llama32_1b"
  for benchmark in "${TABLE8_BENCHMARKS[@]}"; do
    for variant in "${TABLE8_VARIANTS_LLAMA[@]}"; do
      submit_job \
        "table8" \
        "$seed" \
        "llama_${benchmark}_${variant}" \
        "$ROOT/slurm/pretrained_external_eval_single.slurm" \
        "ALL,BENCHMARK=${benchmark},VARIANT=${variant},OUT_ROOT=${llama_out_root},LLAMA_MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct,EVAL_SEED=${seed},SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED=${seed}"
    done
  done
done

printf "[done] Manifest written to %s\n" "$MANIFEST"
