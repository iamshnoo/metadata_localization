#!/bin/bash
set -euo pipefail

SLURM=/path/to/metacul/slurm/pretrained_localnewsqa_adversarial_grid.slurm
LOG_DIR=/path/to/logs/slurm_logs
TS=$(date +%Y%m%d_%H%M%S)
MANIFEST=$LOG_DIR/figure9_adversarial_3b_c1_repairs_${TS}.tsv

mkdir -p "$LOG_DIR"
printf 'jobid\tout_dir\trun_tag\trate_list\tmode\n' > "$MANIFEST"

submit_job() {
  local out_dir="$1"
  local run_tag="$2"
  local mode="$3"

  local jobid
  jobid=$(sbatch --parsable \
    --export=ALL,MODEL_TYPE=custom,META_TAG=with_metadata,CUSTOM_MODEL_PATH_WITH=/path/to/metacul/models/combined_with_metadata_3b,CUSTOM_MODEL_PATH_WITHOUT=/path/to/metacul/models/combined_without_metadata_3b,OUT_DIR="$out_dir",RUN_TAG="$run_tag",RATE_LIST='1.0',URL_CORRUPTION_MODE="$mode",URL_MASK_TOKEN=xx,SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED=42,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,METADATA_PROMPT_STYLE=name_grounded,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=final_answer_colon,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.5,LENGTH_NORM_ALPHA=0.25 \
    "$SLURM")

  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$jobid" "$out_dir" "$run_tag" "1.0" "$mode" >> "$MANIFEST"
}

submit_job \
  /path/to/metacul/results/downstream_localnewsqa_pretrained_adversarial_figure9_aligned/3b \
  3b_tplus_adv \
  full_mismatch

submit_job \
  /path/to/metacul/results/downstream_localnewsqa_pretrained_adversarial_figure9_urlmask/3b \
  3b_tplus_urlmask \
  url_country_mask

echo "$MANIFEST"
