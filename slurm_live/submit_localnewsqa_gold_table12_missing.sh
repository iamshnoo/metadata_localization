#!/usr/bin/env bash
set -euo pipefail

ROOT=/path/to/metacul
DATASET=${DATASET:-$ROOT/qa_data/localnewsqa_core/final_gold_20260516/localnewsqa_gold_explicit17000_ambiguous1700.jsonl}
RUN_ROOT=${RUN_ROOT:-$ROOT/results/localnewsqa_gold_20260516}
WRAPPER=${WRAPPER:-$ROOT/slurm/localnewsqa_eval_baseline_single.slurm}
POSTPROCESS=${POSTPROCESS:-$ROOT/slurm/postprocess_localnewsqa_gold_table12.sh}
LOG_DIR=${LOG_DIR:-/path/to/logs/slurm_logs}
SEED=${SEED:-41}
TIME_LIMIT=${TIME_LIMIT:-06:00:00}
SBATCH_RETRY_SLEEP=${SBATCH_RETRY_SLEEP:-180}
MIN_TARGET_LINES=${MIN_TARGET_LINES:-18700}
MIN_CONTRAST_LINES=${MIN_CONTRAST_LINES:-1700}

mkdir -p "$LOG_DIR" "$RUN_ROOT/external_target/seed_${SEED}" "$RUN_ROOT/external_contrast/seed_${SEED}"
chmod +x "$POSTPROCESS"

STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="$LOG_DIR/localnewsqa_gold_table12_missing_${STAMP}.tsv"
printf 'job_id\taccount\tpartition\tqos\tgres\tmodel_slug\tmodel_name\tmeta_tag\trole\tout_jsonl\n' > "$MANIFEST"

# slug|model_id|preferred_size
MODELS=(
  "llama32_1b_base|meta-llama/Llama-3.2-1B|small"
  "llama32_3b_base|meta-llama/Llama-3.2-3B|medium"
  "qwen25_1p5b_base|Qwen/Qwen2.5-1.5B|small"
  "qwen25_3b_base|Qwen/Qwen2.5-3B|medium"
  "qwen35_0p8b_base|Qwen/Qwen3.5-0.8B-Base|small"
  "qwen35_2b_base|Qwen/Qwen3.5-2B-Base|medium"
  "qwen35_4b_base|Qwen/Qwen3.5-4B-Base|large"
  "gemma4_e2b_base|google/gemma-4-E2B|medium"
  "gemma4_e4b_base|google/gemma-4-E4B|large"
  "ministral3_3b_base|mistralai/Ministral-3-3B-Base-2512|large"
  "ministral3_3b_chat|mistralai/Ministral-3-3B-Instruct-2512|large"
)

# account|partition|qos|gres|mem|python
PROFILES=(
  "cs678sp23|contrib-B200|gpu|gpu:B200.180gb:1|128GB|/path/to/nanotron-b200/bin/python"
  "cs678sp23|gpuq|gpu|gpu:A100.80gb:1|96GB|"
  "cs678sp23|contrib-gpuq|gpu|gpu:A100.80gb:1|96GB|"
  "SLURM_ACCOUNT|contrib-gpuq|gpu|gpu:A100.80gb:1|96GB|"
  "SLURM_ACCOUNT|contrib-gpuq|cs_dept|gpu:A100.80gb:1|96GB|"
)

line_count() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    printf '0\n'
    return
  fi
  wc -l < "$path"
}

is_complete() {
  local path="$1"
  local min_lines="$2"
  local observed
  observed=$(line_count "$path")
  [[ "$observed" -ge "$min_lines" ]]
}

submit_sbatch() {
  local output status
  while true; do
    set +e
    output=$(sbatch --parsable "$@" 2>&1)
    status=$?
    set -e
    if [[ "$status" -eq 0 ]]; then
      printf '%s\n' "$output" | cut -d';' -f1
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

choose_profile_index() {
  local size="$1"
  local counter="$2"
  if [[ "$size" == "large" ]]; then
    printf '%s\n' $((counter % ${#PROFILES[@]}))
  elif [[ "$size" == "medium" ]]; then
    printf '%s\n' $(((counter + 1) % ${#PROFILES[@]}))
  else
    printf '%s\n' $(((counter + 2) % ${#PROFILES[@]}))
  fi
}

JOB_IDS=()
counter=0

for entry in "${MODELS[@]}"; do
  IFS='|' read -r model_slug model_name model_size <<< "$entry"
  for meta_tag in with_metadata without_metadata; do
    for role in target contrast; do
      out_dir="$RUN_ROOT/external_${role}/seed_${SEED}"
      out_jsonl="$out_dir/localnewsqa_${model_slug}_${meta_tag}_${role}.jsonl"
      if [[ "$role" == "target" ]]; then
        min_lines="$MIN_TARGET_LINES"
        split_filter="all"
      else
        min_lines="$MIN_CONTRAST_LINES"
        split_filter="ambiguous"
      fi
      if is_complete "$out_jsonl" "$min_lines"; then
        printf '[skip complete] %s\n' "$out_jsonl"
        continue
      fi

      profile_idx=$(choose_profile_index "$model_size" "$counter")
      IFS='|' read -r account partition qos gres mem python_override <<< "${PROFILES[$profile_idx]}"
      job_name="lnqa-t12-${model_slug}-${meta_tag}-${role}-s${SEED}"
      export_vars="ALL,DATASET=${DATASET},MODEL_NAME=${model_name},MODEL_SLUG=${model_slug},META_TAG=${meta_tag},LOCALE_ROLE=${role},SPLIT_TYPE_FILTER=${split_filter},OUT_DIR=${out_dir},SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED=${SEED},MCQ_SCORING=option_text_avg,ANSWER_FORMAT=option,RESUME=1"
      if [[ -n "$python_override" ]]; then
        export_vars="${export_vars},PYTHON=${python_override}"
      fi
      if [[ "$model_name" == meta-llama/* ]]; then
        export_vars="${export_vars},LOCALIZE_MODEL=1"
      fi

      job_id=$(submit_sbatch \
        --account="$account" \
        --partition="$partition" \
        --qos="$qos" \
        --gres="$gres" \
        --mem="$mem" \
        --time="$TIME_LIMIT" \
        --job-name="$job_name" \
        --export="$export_vars" \
        "$WRAPPER")
      JOB_IDS+=("$job_id")
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$job_id" "$account" "$partition" "$qos" "$gres" "$model_slug" "$model_name" "$meta_tag" "$role" "$out_jsonl" >> "$MANIFEST"
      printf '[submitted] %s %s %s job=%s account=%s partition=%s gres=%s\n' \
        "$model_slug" "$meta_tag" "$role" "$job_id" "$account" "$partition" "$gres"
      counter=$((counter + 1))
    done
  done
done

POST_JOB=""
if ((${#JOB_IDS[@]} > 0)); then
  dep=$(IFS=:; echo "${JOB_IDS[*]}")
  POST_JOB=$(submit_sbatch \
    --dependency=afterok:"$dep" \
    --account=SLURM_ACCOUNT \
    --partition=normal \
    --qos=normal \
    --mem=16GB \
    --time=03:00:00 \
    --job-name=lnqa-t12-post \
    --output="$LOG_DIR/lnqa_t12_post.%j.out" \
    --error="$LOG_DIR/lnqa_t12_post.%j.err" \
    "$POSTPROCESS")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$POST_JOB" "SLURM_ACCOUNT" "normal" "normal" "" "postprocess" "" "" "" "$POSTPROCESS" >> "$MANIFEST"
  printf '[submitted postprocess] job=%s dependency_count=%s\n' "$POST_JOB" "${#JOB_IDS[@]}"
else
  "$POSTPROCESS"
fi

printf '[done] manifest=%s\n' "$MANIFEST"
printf '[done] eval_jobs=%s postprocess_job=%s\n' "${#JOB_IDS[@]}" "${POST_JOB:-ran_inline}"
