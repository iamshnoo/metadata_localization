#!/bin/bash
set -euo pipefail

ROOT=/path/to/metacul
LOG_DIR=/path/to/logs/slurm_logs
mkdir -p "$LOG_DIR"

FIGURE9_MANIFEST="$LOG_DIR/figure9_raw_code_grounded_multiseed_20260418_071920.tsv"
TABLE8_MANIFEST="$LOG_DIR/table8_maple_raw_code_grounded_20260418_073151.tsv"
ADVERSARIAL_MANIFEST="$LOG_DIR/localnewsqa_adversarial_raw_code_grounded_20260418_073212.tsv"

FIGURE9_SLURM="$ROOT/slurm/pretrained_localnewsqa_eval_single.slurm"
TABLE8_SLURM="$ROOT/slurm/pretrained_external_eval_single.slurm"
ADVERSARIAL_SLURM="$ROOT/slurm/pretrained_localnewsqa_adversarial_grid.slurm"

timestamp=$(date +%Y%m%d_%H%M%S)
requeue_manifest="$LOG_DIR/requeued_after_wvs_${timestamp}.tsv"
printf "group\told_jobid\tnew_jobid\n" > "$requeue_manifest"

submit_figure9_row() {
  local old_jobid="$1" seed="$2" size="$3" train_meta="$4" eval_meta="$5" out_dir="$6" run_tag="$7"
  local custom_with="$ROOT/models/combined_with_metadata_${size}"
  local custom_without="$ROOT/models/combined_without_metadata_${size}"
  local metadata_prompt_style="legacy_code"
  if [[ "$eval_meta" == "with_metadata" ]]; then
    metadata_prompt_style="code_grounded"
  fi
  local new_jobid
  new_jobid=$(sbatch --parsable \
    --export=ALL,MODEL_TYPE=custom,META_TAG="$eval_meta",TRAIN_META_TAG="$train_meta",EVAL_META_TAG="$eval_meta",OUT_DIR="$out_dir",RUN_TAG="$run_tag",CUSTOM_MODEL_PATH_WITH="$custom_with",CUSTOM_MODEL_PATH_WITHOUT="$custom_without",SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED="$seed",MCQ_SCORING=option_text_avg,ANSWER_FORMAT=option,METADATA_PROMPT_STYLE="$metadata_prompt_style",QA_PROMPT_STYLE=question \
    "$FIGURE9_SLURM")
  printf "figure9\t%s\t%s\n" "$old_jobid" "$new_jobid" >> "$requeue_manifest"
  echo "requeued figure9 old=$old_jobid new=$new_jobid"
}

submit_table8_row() {
  local old_jobid="$1" seed="$2" size="$3" benchmark="$4" variant="$5" out_root="$6"
  local custom_with="$ROOT/models/combined_with_metadata_${size}"
  local custom_without="$ROOT/models/combined_without_metadata_${size}"
  local new_jobid
  new_jobid=$(sbatch --parsable \
    --export=ALL,BENCHMARK="$benchmark",VARIANT="$variant",OUT_ROOT="$out_root",CUSTOM_MODEL_PATH_WITH="$custom_with",CUSTOM_MODEL_PATH_WITHOUT="$custom_without",EVAL_SEED="$seed",SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED="$seed",METADATA_PROMPT_STYLE=code_grounded \
    "$TABLE8_SLURM")
  printf "table8\t%s\t%s\n" "$old_jobid" "$new_jobid" >> "$requeue_manifest"
  echo "requeued table8 old=$old_jobid new=$new_jobid"
}

submit_adversarial_row() {
  local old_jobid="$1" size="$2" train_meta="$3" out_dir="$4" run_tag="$5"
  local custom_with="$ROOT/models/combined_with_metadata_${size}"
  local custom_without="$ROOT/models/combined_without_metadata_${size}"
  local new_jobid
  new_jobid=$(sbatch --parsable \
    --export=ALL,MODEL_TYPE=custom,META_TAG="$train_meta",OUT_DIR="$out_dir",RUN_TAG="$run_tag",CUSTOM_MODEL_PATH_WITH="$custom_with",CUSTOM_MODEL_PATH_WITHOUT="$custom_without",METADATA_PROMPT_STYLE=code_grounded,QA_PROMPT_STYLE=question \
    "$ADVERSARIAL_SLURM")
  printf "adversarial\t%s\t%s\n" "$old_jobid" "$new_jobid" >> "$requeue_manifest"
  echo "requeued adversarial old=$old_jobid new=$new_jobid"
}

while IFS=$'\t' read -r old_jobid seed size train_meta eval_meta out_dir run_tag; do
  [[ "$old_jobid" == "jobid" ]] && continue
  [[ "$old_jobid" -lt 7004028 || "$old_jobid" -gt 7004056 ]] && continue
  submit_figure9_row "$old_jobid" "$seed" "$size" "$train_meta" "$eval_meta" "$out_dir" "$run_tag"
done < "$FIGURE9_MANIFEST"

while IFS=$'\t' read -r old_jobid seed size benchmark variant out_root; do
  [[ "$old_jobid" == "jobid" ]] && continue
  submit_table8_row "$old_jobid" "$seed" "$size" "$benchmark" "$variant" "$out_root"
done < "$TABLE8_MANIFEST"

while IFS=$'\t' read -r old_jobid size train_meta out_dir run_tag; do
  [[ "$old_jobid" == "jobid" ]] && continue
  submit_adversarial_row "$old_jobid" "$size" "$train_meta" "$out_dir" "$run_tag"
done < "$ADVERSARIAL_MANIFEST"

echo "requeue_manifest=$requeue_manifest"
