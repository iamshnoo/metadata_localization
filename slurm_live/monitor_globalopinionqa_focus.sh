#!/bin/bash
set -euo pipefail

LOG=${1:-/path/to/logs/slurm_logs/globalopinionqa_focus_monitor.log}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-120}

mkdir -p "$(dirname "$LOG")"

while true; do
  {
    echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
    squeue -u USER_NAME -h -o '%i|%j|%T|%M|%r' | rg 'globalopinionqa|bm-globalopinionqa-' || true
    for job_id in $(squeue -u USER_NAME -h -o '%i|%j|%T' | awk -F'|' '$2 ~ /^globalopinionqa/ {print $1}'); do
      out_file="/path/to/logs/slurm_logs/priority_dataset_worker.${job_id}.out"
      echo "--- worker ${job_id} ---"
      if [[ -f "$out_file" ]]; then
        tail -n 12 "$out_file"
      else
        echo "no-log-yet"
      fi
    done
    left=$(squeue -u USER_NAME -h -o '%j' | rg '^globalopinionqa|^bm-globalopinionqa-' | wc -l)
    echo "remaining_globalopinionqa_jobs=${left}"
    echo
  } >> "$LOG" 2>&1

  if [[ "$left" == "0" ]]; then
    break
  fi

  sleep "$INTERVAL_SECONDS"
done
