#!/bin/bash
set -euo pipefail

ROOT=/path/to/metacul
SLURM="$ROOT/slurm/pretrained_external_eval_single.slurm"
LOG_DIR=/path/to/logs/slurm_logs
TS=$(date +%Y%m%d_%H%M%S)
MANIFEST=$LOG_DIR/external_raw3b_legacy_crossbench_seed41_${TS}.tsv

mkdir -p "$LOG_DIR"
printf 'jobid\tbenchmark\tvariant\tconfig_slug\tout_root\n' > "$MANIFEST"

BENCHMARKS=(geomlama globalopinionqa worldvaluebench)
VARIANTS=(custom_tplus_eplus custom_tminus_eminus)
SEED=41
CONFIG_SLUG=legacy_defaults_shuffle
OUT_ROOT="$ROOT/results/external_benchmarks_raw3b_legacy_crossbench_seed41/$CONFIG_SLUG"
EXPORT_SUFFIX="METADATA_PROMPT_STYLE=legacy_code,QA_PROMPT_STYLE=question,ANSWER_CUE_STYLE=none,OMIT_OPTION_LABELS=0,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0"

submit_job() {
  local benchmark="$1"
  local variant="$2"
  local export_str="ALL,BENCHMARK=${benchmark},VARIANT=${variant},OUT_ROOT=${OUT_ROOT},EVAL_SEED=${SEED},SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED=${SEED},CUSTOM_MODEL_PATH_WITH=${ROOT}/models/combined_with_metadata_3b,CUSTOM_MODEL_PATH_WITHOUT=${ROOT}/models/combined_without_metadata_3b,${EXPORT_SUFFIX}"
  local jobid
  jobid=$(sbatch --parsable --export="$export_str" "$SLURM")
  printf '%s\t%s\t%s\t%s\t%s\n' "$jobid" "$benchmark" "$variant" "$CONFIG_SLUG" "$OUT_ROOT" >> "$MANIFEST"
}

for benchmark in "${BENCHMARKS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    submit_job "$benchmark" "$variant"
  done
done

echo "$MANIFEST"
