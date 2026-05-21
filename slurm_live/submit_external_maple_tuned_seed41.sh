#!/bin/bash
set -euo pipefail

ROOT=/path/to/metacul
SLURM="$ROOT/slurm/pretrained_external_eval_single.slurm"
LOG_DIR=/path/to/logs/slurm_logs
TS=$(date +%Y%m%d_%H%M%S)
MANIFEST=$LOG_DIR/external_maple_tuned_seed41_${TS}.tsv

mkdir -p "$LOG_DIR"
printf 'jobid\ttrack\tfamily\tbenchmark\tvariant\tseed\tout_root\trun_label\n' > "$MANIFEST"

BENCHMARKS=(geomlama globalopinionqa worldvaluebench mmlu)
VARIANTS=(
  custom_tplus_eplus
  custom_tplus_eminus
  custom_tminus_eplus
  custom_tminus_eminus
)
SEED=41

submit_job() {
  local track="$1"
  local family="$2"
  local benchmark="$3"
  local variant="$4"
  local out_root="$5"
  shift 5
  local export_str="ALL,BENCHMARK=${benchmark},VARIANT=${variant},OUT_ROOT=${out_root},EVAL_SEED=${SEED},SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED=${SEED},$*"
  local jobid
  jobid=$(sbatch --parsable --export="$export_str" "$SLURM")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$jobid" "$track" "$family" "$benchmark" "$variant" "$SEED" "$out_root" "${track}_${family}_${benchmark}_${variant}_seed${SEED}" >> "$MANIFEST"
}

for benchmark in "${BENCHMARKS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    submit_job \
      raw \
      1b \
      "$benchmark" \
      "$variant" \
      "$ROOT/results/external_benchmarks_maple_tuned_seed41/raw_1b" \
      CUSTOM_MODEL_PATH_WITH="$ROOT/models/combined_with_metadata_1b",CUSTOM_MODEL_PATH_WITHOUT="$ROOT/models/combined_without_metadata_1b",METADATA_PROMPT_STYLE=code_grounded,QA_PROMPT_STYLE=question,ANSWER_CUE_STYLE=final_answer_colon,OMIT_OPTION_LABELS=0,EXACT_OPTION_TEXT_INSTRUCTION=1,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=1.25,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0

    submit_job \
      raw \
      3b \
      "$benchmark" \
      "$variant" \
      "$ROOT/results/external_benchmarks_maple_tuned_seed41/raw_3b" \
      CUSTOM_MODEL_PATH_WITH="$ROOT/models/combined_with_metadata_3b",CUSTOM_MODEL_PATH_WITHOUT="$ROOT/models/combined_without_metadata_3b",METADATA_PROMPT_STYLE=name_grounded,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.5

    submit_job \
      sft \
      1b_chat \
      "$benchmark" \
      "$variant" \
      "$ROOT/results/external_benchmarks_maple_tuned_seed41/sft_1b_chat" \
      CUSTOM_MODEL_PATH_WITH="$ROOT/models/sft/combined_with_metadata_chat",CUSTOM_MODEL_PATH_WITHOUT="$ROOT/models/sft/combined_without_metadata_chat",METADATA_PROMPT_STYLE=code_grounded,QA_PROMPT_STYLE=question,ANSWER_CUE_STYLE=final_answer_colon,OMIT_OPTION_LABELS=0,EXACT_OPTION_TEXT_INSTRUCTION=1,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=1.25,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0

    submit_job \
      sft \
      3b_best3b_chat \
      "$benchmark" \
      "$variant" \
      "$ROOT/results/external_benchmarks_maple_tuned_seed41/sft_3b_best3b_chat" \
      CUSTOM_MODEL_PATH_WITH="$ROOT/models/sft/combined_with_metadata_3b_best3b_chat",CUSTOM_MODEL_PATH_WITHOUT="$ROOT/models/sft/combined_without_metadata_3b_best3b_chat",METADATA_PROMPT_STYLE=name_grounded,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.5
  done
done

echo "$MANIFEST"
