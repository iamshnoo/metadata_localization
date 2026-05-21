#!/bin/bash
set -euo pipefail

ROOT=/path/to/metacul
LAUNCHER="$ROOT/slurm/submit_external_raw3b_unified_qansweravg_resume_v3.sh"
PAUSE_FILE=${PAUSE_FILE:-$ROOT/slurm/.pause_external_gpuq_resume_v3}
SLEEP_SECONDS=${SLEEP_SECONDS:-90}
MAX_LOOPS=${MAX_LOOPS:-0}
LOOP=0

while true; do
  if [[ -f "$PAUSE_FILE" ]]; then
    echo "[paused] drive_external_unified_qansweravg_resume_v3.sh stopping due to $PAUSE_FILE"
    exit 0
  fi

  ((LOOP+=1))
  echo "[loop $LOOP] $(date +%Y-%m-%dT%H:%M:%S)"
  echo "[jobs]"
  squeue -u USER_NAME -o '%.18i %.15P %.28j %.8T %.10M %.6D %R' | rg 'qavgv3|JOBID' || true
  echo "[launcher]"
  bash "$LAUNCHER"

  if (( MAX_LOOPS > 0 && LOOP >= MAX_LOOPS )); then
    echo "[stop] reached MAX_LOOPS=$MAX_LOOPS"
    exit 0
  fi

  sleep "$SLEEP_SECONDS"
done
