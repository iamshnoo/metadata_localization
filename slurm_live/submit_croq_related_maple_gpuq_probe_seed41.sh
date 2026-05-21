#!/bin/bash
set -euo pipefail

ROOT=/path/to/metacul
SLURM="$ROOT/slurm/pretrained_external_eval_single.slurm"
LOG_DIR=/path/to/logs/slurm_logs
TS=$(date +%Y%m%d_%H%M%S)
MANIFEST=$LOG_DIR/croq_related_maple_gpuq_probe_seed41_${TS}.tsv
MAPLE_PYTHON=/path/to/nanotron-env/bin/python

mkdir -p "$LOG_DIR"
printf 'jobid\tbenchmark\tvariant\tseed\tconfig_slug\tout_root\n' > "$MANIFEST"

if [[ -n "${BENCHMARKS_OVERRIDE:-}" ]]; then
  read -r -a BENCHMARKS <<< "$BENCHMARKS_OVERRIDE"
else
  BENCHMARKS=(blend globalmmlu_cs)
fi
VARIANTS=(custom_tplus_eplus custom_tminus_eminus)
SEED=41
SBATCH_TIME=${SBATCH_TIME:-02:00:00}
SBATCH_PARTITION=${SBATCH_PARTITION:-contrib-gpuq}
SBATCH_QOS=${SBATCH_QOS:-cs_dept}
if [[ -n "${CONFIG_FILTER:-}" ]]; then
  read -r -a CONFIG_FILTERS <<< "$CONFIG_FILTER"
else
  CONFIG_FILTERS=()
fi

submit_job() {
  local benchmark="$1"
  local variant="$2"
  local config_slug="$3"
  local out_root="$4"
  local export_suffix="$5"
  local metadata_tag_mode="correct"
  if [[ "$benchmark" == "globalmmlu" || "$benchmark" == "globalmmlu_cs" ]]; then
    metadata_tag_mode="available_only"
  fi
  local export_str="ALL,BENCHMARK=${benchmark},VARIANT=${variant},OUT_ROOT=${out_root},EVAL_SEED=${SEED},SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED=${SEED},PYTHON=${MAPLE_PYTHON},METADATA_TAG_MODE=${metadata_tag_mode},CUSTOM_MODEL_PATH_WITH=${ROOT}/models/combined_with_metadata_3b,CUSTOM_MODEL_PATH_WITHOUT=${ROOT}/models/combined_without_metadata_3b,${export_suffix}"
  local jobid
  jobid=$(sbatch --parsable --partition="$SBATCH_PARTITION" --qos="$SBATCH_QOS" --time="$SBATCH_TIME" --export="$export_str" "$SLURM")
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$jobid" "$benchmark" "$variant" "$SEED" "$config_slug" "$out_root" >> "$MANIFEST"
}

CONFIG_SPECS=(
  "nameplain_question_countryfinal_letter_qmask05_bos|METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=0,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_letter,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.5"
  "nameplain_qanswer_countryfinal_letter_qmask025_nobos|METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=0,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_letter,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.25"
  "nameplain_qanswer_countryfinal_avg_qmask025_nobos|METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.25"
  "countryfirst_qanswer_countryfinal_avg_qmask025_bos|METADATA_PROMPT_STYLE=country_first_strict,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.25"
  "namegrounded_qanswer_countryfinal_avg_qmask025_bos|METADATA_PROMPT_STYLE=name_grounded,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.25"
)

for benchmark in "${BENCHMARKS[@]}"; do
  for spec in "${CONFIG_SPECS[@]}"; do
    IFS="|" read -r config_slug export_suffix <<< "$spec"
    if (( ${#CONFIG_FILTERS[@]} > 0 )); then
      keep=0
      for wanted in "${CONFIG_FILTERS[@]}"; do
        if [[ "$config_slug" == "$wanted" ]]; then
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
        "$benchmark" \
        "$variant" \
        "$config_slug" \
        "$ROOT/results/external_benchmarks_croq_related_maple_gpuq_probe_seed41/${config_slug}" \
        "$export_suffix"
    done
  done
done

echo "$MANIFEST"
