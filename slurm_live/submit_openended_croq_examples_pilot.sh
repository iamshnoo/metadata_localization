#!/bin/bash
set -euo pipefail

ROOT=/path/to/metacul
SLURM="$ROOT/slurm/openended_geo_eval_single.slurm"
LOG_DIR=/path/to/logs/slurm_logs
TS=$(date +%Y%m%d_%H%M%S)
MANIFEST=$LOG_DIR/openended_croq_examples_pilot_${TS}.tsv

mkdir -p "$LOG_DIR"
printf 'jobid\tfamily\tvariant\tout_dir\n' > "$MANIFEST"

submit_job() {
  local family="$1"
  local variant="$2"
  local out_root="$3"
  shift 3
  local export_str="ALL,BENCHMARK=croq_jsonl,VARIANT=${variant},OUT_ROOT=${out_root},INPUT_PATH=${ROOT}/data/openended/croq_paper_examples_en.jsonl,PROMPT_STYLE=croq,EVAL_SEED=41,$*"
  local jobid
  jobid=$(sbatch --parsable --export="$export_str" "$SLURM")
  printf '%s\t%s\t%s\t%s\n' "$jobid" "$family" "$variant" "$out_root" >> "$MANIFEST"
}

submit_job \
  maple_chat_1b \
  custom_tplus_eplus \
  "$ROOT/results/openended_geo_eval_pilot/maple_chat_1b_tplus" \
  CUSTOM_MODEL_PATH_WITH=/path/to/metacul/models/sft/combined_with_metadata_chat,CUSTOM_MODEL_PATH_WITHOUT=/path/to/metacul/models/sft/combined_without_metadata_chat,METADATA_PROMPT_STYLE=name_grounded,MAX_NEW_TOKENS=96,TEMPERATURE=0.0,TOP_P=1.0,DECODING=greedy

submit_job \
  maple_chat_1b \
  custom_tminus_eminus \
  "$ROOT/results/openended_geo_eval_pilot/maple_chat_1b_tminus" \
  CUSTOM_MODEL_PATH_WITH=/path/to/metacul/models/sft/combined_with_metadata_chat,CUSTOM_MODEL_PATH_WITHOUT=/path/to/metacul/models/sft/combined_without_metadata_chat,METADATA_PROMPT_STYLE=name_grounded,MAX_NEW_TOKENS=96,TEMPERATURE=0.0,TOP_P=1.0,DECODING=greedy

submit_job \
  maple_chat_3b \
  custom_tplus_eplus \
  "$ROOT/results/openended_geo_eval_pilot/maple_chat_3b_tplus" \
  CUSTOM_MODEL_PATH_WITH=/path/to/metacul/models/sft/combined_with_metadata_3b_best3b_chat,CUSTOM_MODEL_PATH_WITHOUT=/path/to/metacul/models/sft/combined_without_metadata_3b_best3b_chat,METADATA_PROMPT_STYLE=name_grounded,MAX_NEW_TOKENS=96,TEMPERATURE=0.0,TOP_P=1.0,DECODING=greedy

submit_job \
  maple_chat_3b \
  custom_tminus_eminus \
  "$ROOT/results/openended_geo_eval_pilot/maple_chat_3b_tminus" \
  CUSTOM_MODEL_PATH_WITH=/path/to/metacul/models/sft/combined_with_metadata_3b_best3b_chat,CUSTOM_MODEL_PATH_WITHOUT=/path/to/metacul/models/sft/combined_without_metadata_3b_best3b_chat,METADATA_PROMPT_STYLE=name_grounded,MAX_NEW_TOKENS=96,TEMPERATURE=0.0,TOP_P=1.0,DECODING=greedy

submit_job \
  hf_llama32_1b \
  hf_chat_with_metadata \
  "$ROOT/results/openended_geo_eval_pilot/hf_llama32_1b_tplus" \
  HF_MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct,METADATA_PROMPT_STYLE=name_grounded,MAX_NEW_TOKENS=96,TEMPERATURE=0.0,TOP_P=1.0,DECODING=greedy

submit_job \
  hf_llama32_1b \
  hf_chat_without_metadata \
  "$ROOT/results/openended_geo_eval_pilot/hf_llama32_1b_tminus" \
  HF_MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct,METADATA_PROMPT_STYLE=name_grounded,MAX_NEW_TOKENS=96,TEMPERATURE=0.0,TOP_P=1.0,DECODING=greedy

echo "$MANIFEST"
