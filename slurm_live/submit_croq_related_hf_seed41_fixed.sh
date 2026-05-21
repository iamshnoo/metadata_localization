#!/bin/bash
set -euo pipefail

ROOT=/path/to/metacul
SLURM="$ROOT/slurm/pretrained_external_eval_single.slurm"
LOG_DIR=/path/to/logs/slurm_logs
TS=$(date +%Y%m%d_%H%M%S)
MANIFEST=$LOG_DIR/croq_related_hf_seed41_fixed_${TS}.tsv

mkdir -p "$LOG_DIR"
printf 'jobid\tmodel_key\tmodel_name\tbenchmark\tvariant\tseed\tout_root\n' > "$MANIFEST"

if [[ -n "${BENCHMARKS_OVERRIDE:-}" ]]; then
  read -r -a BENCHMARKS <<< "$BENCHMARKS_OVERRIDE"
else
  BENCHMARKS=(blend globalmmlu_cs normad)
fi
if [[ -n "${MODELS_OVERRIDE:-}" ]]; then
  read -r -a MODELS_FILTER <<< "$MODELS_OVERRIDE"
else
  MODELS_FILTER=()
fi
VARIANTS=(llama3_chat_with_metadata llama3_chat_without_metadata)
SEED=41
SBATCH_TIME=${SBATCH_TIME:-02:30:00}

MODELS=(
  "llama32_3b|meta-llama/Llama-3.2-3B-Instruct"
  "qwen35_0p8b|Qwen/Qwen3.5-0.8B"
  "qwen35_2b|Qwen/Qwen3.5-2B"
  "qwen35_4b|Qwen/Qwen3.5-4B"
  "qwen35_9b|Qwen/Qwen3.5-9B"
  "gemma4_e2b_it|google/gemma-4-E2B-it"
  "gemma4_e4b_it|google/gemma-4-E4B-it"
)

submit_job() {
  local model_key="$1"
  local model_name="$2"
  local benchmark="$3"
  local variant="$4"
  local out_root="$5"
  shift 5
  local metadata_tag_mode="correct"
  if [[ "$benchmark" == "globalmmlu" || "$benchmark" == "globalmmlu_cs" ]]; then
    metadata_tag_mode="available_only"
  fi
  local export_str="ALL,BENCHMARK=${benchmark},VARIANT=${variant},OUT_ROOT=${out_root},EVAL_SEED=${SEED},LLAMA_MODEL_NAME=${model_name},SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED=${SEED},METADATA_TAG_MODE=${metadata_tag_mode},$*"
  local jobid
  jobid=$(sbatch --parsable --time="$SBATCH_TIME" --export="$export_str" "$SLURM")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$jobid" "$model_key" "$model_name" "$benchmark" "$variant" "$SEED" "$out_root" >> "$MANIFEST"
}

for benchmark in "${BENCHMARKS[@]}"; do
  for spec in "${MODELS[@]}"; do
    IFS="|" read -r model_key model_name <<< "$spec"
    if (( ${#MODELS_FILTER[@]} > 0 )); then
      keep=0
      for wanted in "${MODELS_FILTER[@]}"; do
        if [[ "$model_key" == "$wanted" ]]; then
          keep=1
          break
        fi
      done
      if [[ "$keep" != "1" ]]; then
        continue
      fi
    fi
    for variant in "${VARIANTS[@]}"; do
      submit_job \
        "$model_key" \
        "$model_name" \
        "$benchmark" \
        "$variant" \
        "$ROOT/results/external_benchmarks_croq_related_hf_seed41_fixed/${model_key}" \
        METADATA_PROMPT_STYLE=country_first_strict,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.5
    done
  done
done

echo "$MANIFEST"
