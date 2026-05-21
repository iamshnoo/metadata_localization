#!/bin/bash
set -euo pipefail

ROOT=/path/to/metacul
SLURM="$ROOT/slurm/pretrained_external_eval_single.slurm"
LOG_DIR=/path/to/logs/slurm_logs
TS=$(date +%Y%m%d_%H%M%S)
MANIFEST=$LOG_DIR/croq_related_maple_protocol_search_seed41_${TS}.tsv
MAPLE_PYTHON=/path/to/nanotron-env/bin/python

mkdir -p "$LOG_DIR"
printf 'jobid\tfamily\tbenchmark\tvariant\tseed\tconfig_slug\tout_root\n' > "$MANIFEST"

if [[ -n "${BENCHMARKS_OVERRIDE:-}" ]]; then
  read -r -a BENCHMARKS <<< "$BENCHMARKS_OVERRIDE"
else
  BENCHMARKS=(blend globalmmlu_cs)
fi
VARIANTS=(custom_tplus_eplus custom_tminus_eminus)
SEED=41
ENABLE_RAW1B=${ENABLE_RAW1B:-0}
SBATCH_TIME=${SBATCH_TIME:-02:00:00}
if [[ -n "${CONFIG_FILTER:-}" ]]; then
  read -r -a CONFIG_FILTERS <<< "$CONFIG_FILTER"
else
  CONFIG_FILTERS=()
fi

submit_job() {
  local family="$1"
  local benchmark="$2"
  local variant="$3"
  local config_slug="$4"
  local out_root="$5"
  local export_suffix="$6"
  local custom_with="$7"
  local custom_without="$8"
  local metadata_tag_mode="correct"
  if [[ "$benchmark" == "globalmmlu" || "$benchmark" == "globalmmlu_cs" ]]; then
    metadata_tag_mode="available_only"
  fi
  local export_str="ALL,BENCHMARK=${benchmark},VARIANT=${variant},OUT_ROOT=${out_root},EVAL_SEED=${SEED},SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED=${SEED},METADATA_TAG_MODE=${metadata_tag_mode},CUSTOM_MODEL_PATH_WITH=${custom_with},CUSTOM_MODEL_PATH_WITHOUT=${custom_without},${export_suffix}"
  local jobid
  jobid=$(sbatch --parsable --time="$SBATCH_TIME" --export="$export_str" "$SLURM")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$jobid" "$family" "$benchmark" "$variant" "$SEED" "$config_slug" "$out_root" >> "$MANIFEST"
}

RAW1B_WITH="$ROOT/models/combined_with_metadata_1b"
RAW1B_WITHOUT="$ROOT/models/combined_without_metadata_1b"
RAW3B_WITH="$ROOT/models/combined_with_metadata_3b"
RAW3B_WITHOUT="$ROOT/models/combined_without_metadata_3b"

CONFIG_SPECS=(
  "nameplain_question_countryfinal_letter_qmask05_bos|PYTHON=${MAPLE_PYTHON},METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=0,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_letter,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.5"
  "nameplain_qanswer_countryfinal_letter_qmask025_nobos|PYTHON=${MAPLE_PYTHON},METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=0,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_letter,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.25"
  "nameplain_qanswer_countryfinal_avg_qmask025_nobos|PYTHON=${MAPLE_PYTHON},METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.25"
  "nameplain_question_countryfinal_avg_qmask025_nobos|PYTHON=${MAPLE_PYTHON},METADATA_PROMPT_STYLE=name_plain,QA_PROMPT_STYLE=question,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.25"
  "countryfirst_qanswer_countryfinal_avg_qmask025_bos|PYTHON=${MAPLE_PYTHON},METADATA_PROMPT_STYLE=country_first_strict,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.25"
  "countryfirst_question_countryfinal_avg_qmask025_bos|PYTHON=${MAPLE_PYTHON},METADATA_PROMPT_STYLE=country_first_strict,QA_PROMPT_STYLE=question,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.25"
  "namegrounded_qanswer_countryfinal_avg_qmask025_bos|PYTHON=${MAPLE_PYTHON},METADATA_PROMPT_STYLE=name_grounded,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.25"
  "countryfirst_qanswer_countrycorrect_avg_qmask025_bos|PYTHON=${MAPLE_PYTHON},METADATA_PROMPT_STYLE=country_first_strict,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_the_correct_answer_is,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.25"
  "namegrounded_qanswer_countrycorrect_avg_qmask025_bos|PYTHON=${MAPLE_PYTHON},METADATA_PROMPT_STYLE=name_grounded,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_the_correct_answer_is,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.25"
  "countryfirst_qanswer_countryfinal_avg_keepmeta_qmask025_bos|PYTHON=${MAPLE_PYTHON},METADATA_PROMPT_STYLE=country_first_strict,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=question_masked_no_metadata,NULL_CALIBRATION_BETA=0.25"
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
      if [[ "$ENABLE_RAW1B" == "1" ]]; then
        submit_job \
          raw_1b \
          "$benchmark" \
          "$variant" \
          "$config_slug" \
          "$ROOT/results/external_benchmarks_croq_related_maple_protocol_search_seed41_v2/raw_1b/${config_slug}" \
          "$export_suffix" \
          "$RAW1B_WITH" \
          "$RAW1B_WITHOUT"
      fi

      submit_job \
        raw_3b \
        "$benchmark" \
        "$variant" \
        "$config_slug" \
        "$ROOT/results/external_benchmarks_croq_related_maple_protocol_search_seed41_v2/raw_3b/${config_slug}" \
        "$export_suffix" \
        "$RAW3B_WITH" \
        "$RAW3B_WITHOUT"
    done
  done
done

echo "$MANIFEST"
