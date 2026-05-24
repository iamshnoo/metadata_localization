#!/bin/bash
#SBATCH --job-name=lnqa_jam1000
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=localnewsqa-%x-%j.out
#SBATCH --error=localnewsqa-%x-%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
RUN_ROOT=${LOCALNEWSQA_RUN_ROOT:-${REPO_ROOT}/qa_data/localnewsqa_core/runs}
cd "${REPO_ROOT}"
mkdir -p "${RUN_ROOT}/quality_audit/no_api_splits_20260515/web_semantic_gold_ambiguous_1700/explicit_max_audit/strict_defensible_1000_curated_final/slurm_logs"
export PYTHONUNBUFFERED=1

echo "host=$(hostname)"
echo "started_at=$(date -Is)"
python qa_data/localnewsqa_core/65_finalize_explicit_strict_defensible_1000_jamaica.py
echo "finished_at=$(date -Is)"
