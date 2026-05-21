#!/bin/bash
set -euo pipefail

ROOT=/path/to/metacul
SLURM="$ROOT/slurm/pretrained_external_eval_packed.slurm"
LOG_DIR=/path/to/logs/slurm_logs
ANALYSIS_DIR="$ROOT/results/analysis/external_raw3b_unified_search_round6"
RESULT_ROOT="$ROOT/results/external_benchmarks_raw3b_unified_search_round6"
PYTHON=/path/to/nanotron-b200/bin/python
DISPATCH_SCRIPT="$ROOT/slurm/dispatch_external_unified_followup.sh"
TS=$(date +%Y%m%d_%H%M%S)
MANIFEST=$LOG_DIR/external_raw3b_unified_search_round6_${TS}.tsv

mkdir -p "$LOG_DIR" "$ANALYSIS_DIR" "$RESULT_ROOT"
printf 'jobid\tconfig_slug\tout_root\n' > "$MANIFEST"

BENCHMARKS=geomlama:globalopinionqa:worldvaluebench:mmlu
VARIANTS=custom_tplus_eplus:custom_tminus_eminus
SEED=41
JOB_IDS=()

submit_config_job() {
  local config_slug="$1"
  local export_suffix="$2"
  local out_root="$RESULT_ROOT/$config_slug"
  local export_str="ALL,PYTHON=${PYTHON},BENCHMARKS=${BENCHMARKS},VARIANTS=${VARIANTS},OUT_ROOT=${out_root},EVAL_SEED=${SEED},SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED=${SEED},CUSTOM_MODEL_PATH_WITH=${ROOT}/models/combined_with_metadata_3b,CUSTOM_MODEL_PATH_WITHOUT=${ROOT}/models/combined_without_metadata_3b,${export_suffix}"
  local jobid
  jobid=$(sbatch --parsable --export="$export_str" "$SLURM")
  JOB_IDS+=("$jobid")
  printf '%s\t%s\t%s\n' "$jobid" "$config_slug" "$out_root" >> "$MANIFEST"
}

declare -A CONFIG_EXPORTS

CONFIG_EXPORTS[name_plain_qanswer_countryfinal_labels_qmask025_nobos]="METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=0,EXACT_OPTION_TEXT_INSTRUCTION=1,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.25"
CONFIG_EXPORTS[name_plain_question_countryfinal_labels_qmask025_nobos]="METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=0,EXACT_OPTION_TEXT_INSTRUCTION=1,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.25"
CONFIG_EXPORTS[name_plain_qanswer_countryfinal_letter_qmask025_nobos]="METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=0,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_letter,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.25"
CONFIG_EXPORTS[name_plain_question_countryfinal_letter_qmask025_nobos]="METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=0,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_letter,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.25"

for config_slug in "${!CONFIG_EXPORTS[@]}"; do
  submit_config_job "$config_slug" "${CONFIG_EXPORTS[$config_slug]}"
done

if ((${#JOB_IDS[@]} > 0)); then
  DEP=$(IFS=:; echo "${JOB_IDS[*]}")
  SUMMARY_JOBID=$(sbatch --parsable \
    --dependency=afterok:${DEP} \
    --job-name=ext-proto-sum \
    --output="$LOG_DIR/ext_proto_sum_round6.%j.out" \
    --error="$LOG_DIR/ext_proto_sum_round6.%j.err" \
    --wrap="$PYTHON $ROOT/src/59_external_unified_protocol_summary.py --root $ROOT/results/external_benchmarks_raw3b_unified_search_round6 --output-csv $ANALYSIS_DIR/detailed.csv --comparison-csv $ANALYSIS_DIR/comparison.csv --min-rows 150")
  sbatch --parsable \
    --dependency=afterok:${SUMMARY_JOBID} \
    --job-name=ext-follow6 \
    --output="$LOG_DIR/ext_follow6.%j.out" \
    --error="$LOG_DIR/ext_follow6.%j.err" \
    --export="ALL,CURRENT_ROOT=$RESULT_ROOT,CURRENT_CONFIGS=name_plain_qanswer_countryfinal_labels_qmask025_nobos,name_plain_question_countryfinal_labels_qmask025_nobos,name_plain_qanswer_countryfinal_letter_qmask025_nobos,name_plain_question_countryfinal_letter_qmask025_nobos,NEXT_SCRIPT=$ROOT/slurm/submit_external_raw3b_unified_search_round5.sh,NEXT_ROOT=$ROOT/results/external_benchmarks_raw3b_unified_search_round5" \
    --wrap="bash $DISPATCH_SCRIPT"
fi

echo "$MANIFEST"
