#!/bin/bash
set -euo pipefail

ROOT=/path/to/metacul
PYTHON=/path/to/nanotron-b200/bin/python
GATE="$ROOT/src/60_external_unified_gate.py"

: "${CURRENT_ROOT:?CURRENT_ROOT must be set}"
: "${CURRENT_CONFIGS:?CURRENT_CONFIGS must be set}"
: "${NEXT_SCRIPT:?NEXT_SCRIPT must be set}"

IFS=',' read -r -a CONFIGS <<< "$CURRENT_CONFIGS"

NEXT_ROOT=${NEXT_ROOT:-}
if [[ -z "$NEXT_ROOT" ]]; then
  inferred_next_root=$(rg -o '\$ROOT/results/external_benchmarks[^"/ ]+' "$NEXT_SCRIPT" | head -1 || true)
  if [[ -n "$inferred_next_root" ]]; then
    NEXT_ROOT=${inferred_next_root/\$ROOT/$ROOT}
  fi
fi

if "$PYTHON" "$GATE" --root "$CURRENT_ROOT" --configs "${CONFIGS[@]}"; then
  echo "[ok] current root already satisfies all constraints; no follow-up submission needed."
  exit 0
fi

if [[ -n "$NEXT_ROOT" && -d "$NEXT_ROOT" ]]; then
  echo "[info] follow-up root already exists; skipping duplicate submission: $NEXT_ROOT"
  exit 0
fi

echo "[info] current root did not satisfy all constraints; submitting follow-up wave: $NEXT_SCRIPT"
bash "$NEXT_SCRIPT"
