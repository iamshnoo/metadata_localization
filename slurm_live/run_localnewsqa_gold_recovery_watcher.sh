#!/usr/bin/env bash
set -euo pipefail

ROOT=/path/to/metacul
PYTHON=${PYTHON:-/path/to/nanotron-b200/bin/python}
RUN_ROOT=${RUN_ROOT:-$ROOT/results/localnewsqa_gold_20260516}
LOG_PREFIX="[lnqa-gold-recover]"
WAIT_SECONDS=${WAIT_SECONDS:-604800}
POLL_SECONDS=${POLL_SECONDS:-300}

CURRENT_MANIFEST=${CURRENT_MANIFEST:-/path/to/logs/slurm_logs/localnewsqa_gold_reruns_20260516_062858.tsv}
PREVIOUS_MANIFEST=${PREVIOUS_MANIFEST:-/path/to/logs/slurm_logs/localnewsqa_gold_valid_non1bpretrain_20260516.tsv}

PRETRAIN_TARGET_ROOT="$RUN_ROOT/pretrained_target"
PRETRAIN_CONTRAST_ROOT="$RUN_ROOT/pretrained_contrast"
SFT_TARGET_ROOT="$RUN_ROOT/sft_target"
SFT_CONTRAST_ROOT="$RUN_ROOT/sft_contrast"
EXTERNAL_TARGET_ROOT="$RUN_ROOT/external_target"
EXTERNAL_CONTRAST_ROOT="$RUN_ROOT/external_contrast"
ADVERSARIAL_ROOT="$RUN_ROOT/adversarial_full_mismatch"
URLMASK_ROOT="$RUN_ROOT/adversarial_urlmask"
SUMMARY_ROOT="$RUN_ROOT/summaries"
PLOT_ROOT="$RUN_ROOT/plots"

read_job_ids() {
  local manifest="$1"
  awk -F '\t' 'NR > 1 && $1 ~ /^[0-9]+$/ {print $1}' "$manifest"
}

mapfile -t JOB_IDS < <(cat <(read_job_ids "$PREVIOUS_MANIFEST") <(read_job_ids "$CURRENT_MANIFEST") | sort -u)
if ((${#JOB_IDS[@]} == 0)); then
  printf '%s no job ids found in manifests\n' "$LOG_PREFIX" >&2
  exit 1
fi

printf '%s watching %s GPU jobs\n' "$LOG_PREFIX" "${#JOB_IDS[@]}"
deadline=$((SECONDS + WAIT_SECONDS))
while true; do
  status_text=$(sacct -j "$(IFS=,; echo "${JOB_IDS[*]}")" --format=JobID,State -P || true)
  completed=$(awk -F '|' 'NR > 1 && $1 !~ /\./ && $2 == "COMPLETED" {n++} END {print n+0}' <<< "$status_text")
  active=$(awk -F '|' 'NR > 1 && $1 !~ /\./ && ($2 == "PENDING" || $2 == "RUNNING" || $2 == "CONFIGURING" || $2 == "COMPLETING") {n++} END {print n+0}' <<< "$status_text")
  bad=$(awk -F '|' 'NR > 1 && $1 !~ /\./ && $2 != "COMPLETED" && $2 != "PENDING" && $2 != "RUNNING" && $2 != "CONFIGURING" && $2 != "COMPLETING" {print}' <<< "$status_text")

  if [[ -n "$bad" ]]; then
    printf '%s found failed/non-completed GPU jobs:\n%s\n' "$LOG_PREFIX" "$bad" >&2
    exit 1
  fi
  if ((completed == ${#JOB_IDS[@]})); then
    break
  fi
  if ((SECONDS >= deadline)); then
    printf '%s timed out: completed=%s active=%s total=%s\n' "$LOG_PREFIX" "$completed" "$active" "${#JOB_IDS[@]}" >&2
    exit 1
  fi
  printf '%s waiting: completed=%s active=%s total=%s; sleeping %ss\n' "$LOG_PREFIX" "$completed" "$active" "${#JOB_IDS[@]}" "$POLL_SECONDS"
  sleep "$POLL_SECONDS"
done

mkdir -p "$SUMMARY_ROOT" "$PLOT_ROOT"

printf '%s running pretrained summaries\n' "$LOG_PREFIX"
"$PYTHON" "$ROOT/src/50_localnewsqa_pretrained_frozen_multiseed_summary.py" \
  --root "$PRETRAIN_TARGET_ROOT" \
  --plot-csv "$PLOT_ROOT/plot_8_pretrained_target_split_multiseed.csv"
"$PYTHON" "$ROOT/src/68_localnewsqa_seed41_bootstrap_summary.py" \
  --input-root "$PRETRAIN_TARGET_ROOT" \
  --output-csv "$PLOT_ROOT/plot_8_pretrained_target_split_seed41_bootstrap.csv"
"$PYTHON" "$ROOT/src/55_localnewsqa_locale_switch.py" \
  --target-root-1b "$PRETRAIN_TARGET_ROOT/1b_codeg_labels_question_final" \
  --contrast-root-1b "$PRETRAIN_CONTRAST_ROOT/1b_codeg_labels_question_final" \
  --target-root-3b "$PRETRAIN_TARGET_ROOT/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos" \
  --contrast-root-3b "$PRETRAIN_CONTRAST_ROOT/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos" \
  --output-root "$SUMMARY_ROOT/localnewsqa_locale_switch"

printf '%s running SFT summaries\n' "$LOG_PREFIX"
"$PYTHON" "$ROOT/src/71_localnewsqa_gold_sft_summary.py" \
  --sft-root "$SFT_TARGET_ROOT" \
  --plot-csv "$PLOT_ROOT/plot_8_sft_target_split_multiseed_gold.csv"
"$PYTHON" "$ROOT/src/61_localnewsqa_sft_locale_switch.py" \
  --target-root-1b "$SFT_TARGET_ROOT/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos" \
  --contrast-root-1b "$SFT_CONTRAST_ROOT/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos" \
  --target-root-3b "$SFT_TARGET_ROOT/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos" \
  --contrast-root-3b "$SFT_CONTRAST_ROOT/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos" \
  --output-root "$SUMMARY_ROOT/localnewsqa_sft_locale_switch"

printf '%s running external baseline summaries\n' "$LOG_PREFIX"
"$PYTHON" "$ROOT/src/35_localnewsqa_multiseed_summary.py" \
  --pretrained-root "$PRETRAIN_TARGET_ROOT/not_used" \
  --baseline-root "$EXTERNAL_TARGET_ROOT" \
  --plot-csv "$PLOT_ROOT/unused_pretrained_from_external_summary.csv" \
  --table7-long-csv "$PLOT_ROOT/table7_external_localnewsqa_multiseed_long.csv" \
  --table7-wide-csv "$PLOT_ROOT/table7_external_localnewsqa_multiseed_wide.csv"
"$PYTHON" "$ROOT/src/56_localnewsqa_external_locale_switch.py" \
  --target-root "$EXTERNAL_TARGET_ROOT" \
  --contrast-root "$EXTERNAL_CONTRAST_ROOT" \
  --output-root "$SUMMARY_ROOT/localnewsqa_external_locale_switch"

printf '%s running adversarial summaries\n' "$LOG_PREFIX"
"$PYTHON" "$ROOT/src/44_pretrained_adversarial_summary.py" \
  --input-dir "$ADVERSARIAL_ROOT" \
  --output-csv "$PLOT_ROOT/adversarial_pretrained_summary_full_mismatch.csv"
"$PYTHON" "$ROOT/src/44_pretrained_adversarial_summary.py" \
  --input-dir "$URLMASK_ROOT" \
  --output-csv "$PLOT_ROOT/adversarial_pretrained_summary_urlmask.csv"

printf '%s running final postprocess\n' "$LOG_PREFIX"
WAIT_SECONDS=60 POLL_SECONDS=5 "$ROOT/slurm/run_localnewsqa_gold_postprocess.sh"
