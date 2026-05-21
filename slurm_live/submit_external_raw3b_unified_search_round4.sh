#!/bin/bash
set -euo pipefail

ROOT=/path/to/metacul
SLURM="$ROOT/slurm/pretrained_external_eval_single.slurm"
LOG_DIR=/path/to/logs/slurm_logs
ANALYSIS_DIR="$ROOT/results/analysis/external_raw3b_unified_search_round4"
PYTHON=/path/to/nanotron-b200/bin/python
TS=$(date +%Y%m%d_%H%M%S)
MANIFEST=$LOG_DIR/external_raw3b_unified_search_round4_${TS}.tsv

mkdir -p "$LOG_DIR" "$ANALYSIS_DIR"
printf 'jobid\tbenchmark\tvariant\tconfig_slug\tout_root\n' > "$MANIFEST"

BENCHMARKS=(geomlama globalopinionqa worldvaluebench mmlu)
VARIANTS=(custom_tplus_eplus custom_tminus_eminus)
SEED=41
JOB_IDS=()

submit_config_job() {
  local config_slug="$1"
  local export_suffix="$2"
  local benchmark="$3"
  local variant="$4"
  local out_root="$ROOT/results/external_benchmarks_raw3b_unified_search_round4/$config_slug"
  local export_str="ALL,PYTHON=${PYTHON},BENCHMARK=${benchmark},VARIANT=${variant},OUT_ROOT=${out_root},EVAL_SEED=${SEED},SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED=${SEED},CUSTOM_MODEL_PATH_WITH=${ROOT}/models/combined_with_metadata_3b,CUSTOM_MODEL_PATH_WITHOUT=${ROOT}/models/combined_without_metadata_3b,${export_suffix}"
  local jobid
  jobid=$(sbatch --parsable --export="$export_str" "$SLURM")
  JOB_IDS+=("$jobid")
  printf '%s\t%s\t%s\t%s\t%s\n' "$jobid" "$benchmark" "$variant" "$config_slug" "$out_root" >> "$MANIFEST"
}

declare -A CONFIG_EXPORTS

CONFIG_EXPORTS[name_plain_qanswer_countryfinal_qmask025_nobos]="METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.25"
CONFIG_EXPORTS[name_plain_qanswer_countryfinal_qmask025_bos]="METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.25"

for config_slug in "${!CONFIG_EXPORTS[@]}"; do
  for benchmark in "${BENCHMARKS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
      submit_config_job "$config_slug" "${CONFIG_EXPORTS[$config_slug]}" "$benchmark" "$variant"
    done
  done
done

if ((${#JOB_IDS[@]} > 0)); then
  DEP=$(IFS=:; echo "${JOB_IDS[*]}")
  sbatch --parsable \
    --dependency=afterok:${DEP} \
    --job-name=ext-proto-sum \
    --output="$LOG_DIR/ext_proto_sum_round4.%j.out" \
    --error="$LOG_DIR/ext_proto_sum_round4.%j.err" \
    --wrap="$PYTHON $ROOT/src/59_external_unified_protocol_summary.py --root $ROOT/results/external_benchmarks_raw3b_unified_search_round4 --output-csv $ANALYSIS_DIR/detailed.csv --comparison-csv $ANALYSIS_DIR/comparison.csv --min-rows 150"
fi

echo "$MANIFEST"
