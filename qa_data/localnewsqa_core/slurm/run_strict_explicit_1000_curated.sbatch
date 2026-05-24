#!/bin/bash
#SBATCH --job-name=lnqa_exp1000
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=localnewsqa-%x-%j.out
#SBATCH --error=localnewsqa-%x-%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
RUN_ROOT=${LOCALNEWSQA_RUN_ROOT:-${REPO_ROOT}/qa_data/localnewsqa_core/runs}
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1

echo "host=$(hostname)"
echo "started_at=$(date -Is)"
python qa_data/localnewsqa_core/64_build_explicit_strict_defensible_1000_curated.py
echo "finished_at=$(date -Is)"
