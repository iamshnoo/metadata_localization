#!/bin/bash
set -euo pipefail

ROOT=/path/to/metacul
SLURM=${SLURM:-$ROOT/slurm/pretrained_external_eval_packed.slurm}
LOG_DIR=/path/to/logs/slurm_logs
PYTHON=/path/to/nanotron-b200/bin/python
PAUSE_FILE=${PAUSE_FILE:-$ROOT/slurm/.pause_external_gpuq_resume_v3}
SEED=${SEED:-41}
BENCHMARKS_FILTER=${BENCHMARKS_FILTER:-globalopinionqa,mmlu}
MMLU_CACHE=${MMLU_CACHE:-/path/to/workspace/.cache/hf_datasets_mmlu_fresh}
TS=$(date +%Y%m%d_%H%M%S)
MANIFEST=$LOG_DIR/external_raw3b_unified_qansweravg_resume_v3_${TS}.tsv

if [[ -f "$PAUSE_FILE" ]]; then
  echo "[paused] external qansweravg v3 resume submissions disabled by $PAUSE_FILE"
  exit 0
fi

mkdir -p "$LOG_DIR"
printf 'timestamp\tjobid\tjob_name\tbenchmark\tplus_rows\tminus_rows\tout_root\tstatus\n' > "$MANIFEST"

OUT_ROOT="$ROOT/results/external_benchmarks_raw3b_unified_search_round4/name_plain_qanswer_countryfinal_qmask025_nobos"
EXPORT_SUFFIX="METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.25"

declare -A EXPECTED_ROWS=(
  [geomlama]=150
  [globalopinionqa]=10275
  [worldvaluebench]=1405
  [mmlu]=14042
)

declare -A JOB_NAMES=(
  [globalopinionqa]=qavgv3_goqa
  [worldvaluebench]=qavgv3_wvb
  [mmlu]=qavgv3_mmlu
)

want_benchmark() {
  local benchmark="$1"
  local normalized=",${BENCHMARKS_FILTER//:/,},"
  [[ "$normalized" == *",$benchmark,"* ]]
}

jsonl_path() {
  local benchmark="$1"
  local variant="$2"
  echo "$OUT_ROOT/seed_${SEED}/${benchmark}/${benchmark}_${variant}.jsonl"
}

current_rows() {
  local benchmark="$1"
  local variant="$2"
  local path
  path=$(jsonl_path "$benchmark" "$variant")
  if [[ -f "$path" ]]; then
    wc -l < "$path"
  else
    echo 0
  fi
}

job_active() {
  local job_name="$1"
  squeue -h -u USER_NAME -n "$job_name" | grep -q .
}

submit_job() {
  local benchmark="$1"
  local job_name="${JOB_NAMES[$benchmark]}"
  local plus_rows minus_rows variants extra_export export_str jobid
  plus_rows=$(current_rows "$benchmark" "custom_tplus_eplus")
  minus_rows=$(current_rows "$benchmark" "custom_tminus_eminus")

  if (( minus_rows < plus_rows )); then
    variants="custom_tminus_eminus:custom_tplus_eplus"
  else
    variants="custom_tplus_eplus:custom_tminus_eminus"
  fi

  if (( plus_rows >= EXPECTED_ROWS[$benchmark] && minus_rows >= EXPECTED_ROWS[$benchmark] )); then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date +%Y-%m-%dT%H:%M:%S)" "-" "$job_name" "$benchmark" "$plus_rows" "$minus_rows" "$OUT_ROOT" "complete" >> "$MANIFEST"
    echo "[complete] $benchmark plus=$plus_rows/${EXPECTED_ROWS[$benchmark]} minus=$minus_rows/${EXPECTED_ROWS[$benchmark]}"
    return
  fi

  if job_active "$job_name"; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date +%Y-%m-%dT%H:%M:%S)" "-" "$job_name" "$benchmark" "$plus_rows" "$minus_rows" "$OUT_ROOT" "active" >> "$MANIFEST"
    echo "[active] $job_name plus=$plus_rows/${EXPECTED_ROWS[$benchmark]} minus=$minus_rows/${EXPECTED_ROWS[$benchmark]}"
    return
  fi

  extra_export=""
  if [[ "$benchmark" == "mmlu" ]]; then
    extra_export=",HF_DATASETS_CACHE=${MMLU_CACHE}"
  fi

  export_str="ALL,PYTHON=${PYTHON},PAUSE_FILE=${PAUSE_FILE},BENCHMARKS=${benchmark},VARIANTS=${variants},OUT_ROOT=${OUT_ROOT},EVAL_SEED=${SEED},SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED=${SEED},CUSTOM_MODEL_PATH_WITH=${ROOT}/models/combined_with_metadata_3b,CUSTOM_MODEL_PATH_WITHOUT=${ROOT}/models/combined_without_metadata_3b,${EXPORT_SUFFIX}${extra_export}"
  jobid=$(sbatch --parsable --job-name="$job_name" --export="$export_str" "$SLURM")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date +%Y-%m-%dT%H:%M:%S)" "$jobid" "$job_name" "$benchmark" "$plus_rows" "$minus_rows" "$OUT_ROOT" "submitted" >> "$MANIFEST"
  echo "[submit] $jobid $job_name plus=$plus_rows/${EXPECTED_ROWS[$benchmark]} minus=$minus_rows/${EXPECTED_ROWS[$benchmark]} variants=$variants"
}

for benchmark in globalopinionqa worldvaluebench mmlu; do
  if want_benchmark "$benchmark"; then
    submit_job "$benchmark"
  fi
done

echo "$MANIFEST"
