#!/bin/bash
set -euo pipefail

ROOT=/path/to/metacul
SLURM="$ROOT/slurm/pretrained_external_eval_single.slurm"
LOG_DIR=/path/to/logs/slurm_logs
TS=$(date +%Y%m%d_%H%M%S)
MANIFEST=$LOG_DIR/croq_related_seed41_globalmmlu_${TS}.tsv

mkdir -p "$LOG_DIR"
printf 'jobid\tfamily\tbenchmark\tvariant\tseed\tout_root\trun_label\n' > "$MANIFEST"

BENCHMARKS=(globalmmlu)
VARIANTS=(
  custom_tplus_eplus
  custom_tplus_eminus
  custom_tminus_eplus
  custom_tminus_eminus
)
SEED=41

submit_job() {
  local family="$1"
  local benchmark="$2"
  local variant="$3"
  local out_root="$4"
  shift 4
  local export_str="ALL,BENCHMARK=${benchmark},VARIANT=${variant},OUT_ROOT=${out_root},EVAL_SEED=${SEED},SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED=${SEED},$*"
  local jobid
  jobid=$(sbatch --parsable --export="$export_str" "$SLURM")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$jobid" "$family" "$benchmark" "$variant" "$SEED" "$out_root" "${family}_${benchmark}_${variant}_seed${SEED}" >> "$MANIFEST"
}

for benchmark in "${BENCHMARKS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    submit_job \
      raw_1b \
      "$benchmark" \
      "$variant" \
      "$ROOT/results/external_benchmarks_croq_related_seed41_globalmmlu/raw_1b" \
      CUSTOM_MODEL_PATH_WITH="$ROOT/models/combined_with_metadata_1b",CUSTOM_MODEL_PATH_WITHOUT="$ROOT/models/combined_without_metadata_1b",METADATA_PROMPT_STYLE=code_grounded,QA_PROMPT_STYLE=question,ANSWER_CUE_STYLE=final_answer_colon,OMIT_OPTION_LABELS=0,EXACT_OPTION_TEXT_INSTRUCTION=1,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=1.25,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0

    submit_job \
      raw_3b \
      "$benchmark" \
      "$variant" \
      "$ROOT/results/external_benchmarks_croq_related_seed41_globalmmlu/raw_3b" \
      CUSTOM_MODEL_PATH_WITH="$ROOT/models/combined_with_metadata_3b",CUSTOM_MODEL_PATH_WITHOUT="$ROOT/models/combined_without_metadata_3b",METADATA_PROMPT_STYLE=name_grounded,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.5
  done
done

echo "$MANIFEST"
