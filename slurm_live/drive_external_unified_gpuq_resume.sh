#!/bin/bash
set -euo pipefail

if [[ "${DISABLE_DRIVER:-0}" == "1" ]]; then
  echo "[disabled] drive_external_unified_gpuq_resume.sh disabled via DISABLE_DRIVER=1."
  exit 0
fi

ROOT=/path/to/metacul
QUESTION_LAUNCHER="$ROOT/slurm/submit_external_raw3b_unified_question_gpuq_resume.sh"
SUM_LAUNCHER="$ROOT/slurm/submit_external_raw3b_unified_sum_gpuq_resume.sh"
PAUSE_FILE=${PAUSE_FILE:-$ROOT/slurm/.pause_external_gpuq_resume}
SLEEP_SECONDS=${SLEEP_SECONDS:-90}
MAX_LOOPS=${MAX_LOOPS:-0}
LOOP=0

while true; do
  if [[ -f "$PAUSE_FILE" ]]; then
    echo "[paused] drive_external_unified_gpuq_resume.sh stopping due to $PAUSE_FILE"
    exit 0
  fi

  ((LOOP+=1))
  echo "[loop $LOOP] $(date +%Y-%m-%dT%H:%M:%S)"
  echo "[jobs]"
  squeue -u USER_NAME -o '%.18i %.15P %.28j %.8T %.10M %.6D %R' | rg 'questiongpuq|sumgpuq|JOBID' || true
  echo "[question launcher]"
  bash "$QUESTION_LAUNCHER"
  echo "[sum launcher]"
  bash "$SUM_LAUNCHER"

  if (( MAX_LOOPS > 0 && LOOP >= MAX_LOOPS )); then
    echo "[stop] reached MAX_LOOPS=$MAX_LOOPS"
    exit 0
  fi

  sleep "$SLEEP_SECONDS"
done
