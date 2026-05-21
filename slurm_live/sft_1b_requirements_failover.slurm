#!/bin/bash
#SBATCH --job-name=sft1b-failover
#SBATCH --partition=interactive
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/logs/slurm_logs/sft1b_failover.%j.out
#SBATCH --error=/path/to/logs/slurm_logs/sft1b_failover.%j.err

set -euo pipefail

: "${ONE_B_CSV:?ONE_B_CSV must be set}"
: "${BASELINE_CSV:?BASELINE_CSV must be set}"
: "${FAILOVER_CHAIN:?FAILOVER_CHAIN must be set}"

PYTHON=${PYTHON:-/path/to/nanotron-b200/bin/python}

if "$PYTHON" - "$ONE_B_CSV" "$BASELINE_CSV" <<'PYCODE'
import sys

import pandas as pd


def load_accuracy_map(path: str) -> dict[tuple[str, str], float]:
    df = pd.read_csv(path)
    return {
        (str(row["series"]).strip(), str(row["split"]).strip()): float(row["accuracy"])
        for _, row in df.iterrows()
    }


one_b = load_accuracy_map(sys.argv[1])
baseline = load_accuracy_map(sys.argv[2])

ok = True
for split in ("Overall", "Explicit", "Ambiguous"):
    ok = ok and (one_b[("1B T+/I+", split)] > one_b[("1B T-/I-", split)])
    ok = ok and (baseline[("3B T+/I+", split)] > baseline[("3B T-/I-", split)])
    ok = ok and (baseline[("3B T+/I+", split)] > one_b[("1B T+/I+", split)])

raise SystemExit(0 if ok else 1)
PYCODE
then
  echo "[pass] frozen direct path satisfied requirements; no failover launch"
else
  echo "[fail] frozen direct path missed requirements; launching failover chain"
  bash "$FAILOVER_CHAIN"
fi
