#!/bin/bash
set -euo pipefail

LOG_PATH=${1:-/path/to/logs/slurm_logs/priority_benchmark_monitor.log}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-300}

mkdir -p "$(dirname "$LOG_PATH")"

while true; do
  {
    echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') ====="

    python - <<'PY'
import subprocess
from collections import Counter

def get_lines(cmd: str):
    out = subprocess.check_output(["bash", "-lc", cmd], text=True)
    return [line for line in out.splitlines() if line.strip()]

bm_lines = get_lines("squeue -u USER_NAME -h -o '%j|%T|%M|%R' | rg '^bm-|^geomlama-' || true")
counts = Counter()
dataset_counts = Counter()
running = []

for line in bm_lines:
    name, state, elapsed, reason = line.split("|", 3)
    counts[state] += 1
    if name.startswith("bm-"):
        parts = name.split("-")
        if len(parts) > 1:
            dataset_counts[parts[1]] += 1
    if state == "RUNNING":
        running.append((name, elapsed, reason))

print("active_counts", dict(counts))
print("dataset_pending_mix", dict(sorted(dataset_counts.items())))
print("running_jobs")
for name, elapsed, reason in running:
    print(f"  {name} | {elapsed} | {reason}")
PY

    for job_id in $(squeue -u USER_NAME -h -o '%i|%j|%T' | awk -F'|' '$2 ~ /^geomlama-/ {print $1}'); do
      out_file="/path/to/logs/slurm_logs/geomlama_worker.${job_id}.out"
      echo "--- geomlama_worker ${job_id} out tail ---"
      if [[ -f "$out_file" ]]; then
        tail -n 12 "$out_file"
      else
        echo "missing $out_file"
      fi
    done

    for job_id in $(squeue -u USER_NAME -h -o '%i|%j|%T' | awk -F'|' '$2 ~ /^bm-globalopinionqa-/ && $3 == "RUNNING" {print $1}'); do
      out_file="/path/to/logs/slurm_logs/ext_pretrained.${job_id}.out"
      echo "--- globalopinionqa ${job_id} out tail ---"
      if [[ -f "$out_file" ]]; then
        tail -n 10 "$out_file"
      else
        echo "missing $out_file"
      fi
    done

    echo
  } >> "$LOG_PATH" 2>&1

  sleep "$INTERVAL_SECONDS"
done
