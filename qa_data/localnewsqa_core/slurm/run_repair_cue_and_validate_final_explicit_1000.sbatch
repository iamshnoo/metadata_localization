#!/bin/bash
#SBATCH --job-name=lnqa_exp17k_cueval
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=localnewsqa-%x-%j.out
#SBATCH --error=localnewsqa-%x-%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
RUN_ROOT=${LOCALNEWSQA_RUN_ROOT:-${REPO_ROOT}/qa_data/localnewsqa_core/runs}
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1

FINAL_DIR=${RUN_ROOT}/quality_audit/no_api_splits_20260515/web_semantic_gold_ambiguous_1700/explicit_max_audit/strict_defensible_1000_curated_final
FINAL_JSONL=${FINAL_DIR}/localnewsqa_targetqa_explicit_strict_defensible_1000_per_country_curated_final.jsonl
FINAL_FETCH=${FINAL_DIR}/strict_defensible_1000_curated_final_target_evidence_fetches.jsonl
AMBIG=${RUN_ROOT}/quality_audit/no_api_splits_20260515/web_semantic_gold_ambiguous_1700/localnewsqa_ambiguous_semantic_gold_1700.jsonl
AUDIT_DIR=${FINAL_DIR}/audit
VALIDATION_DIR=${FINAL_DIR}/final_validation
LOG_DIR=${FINAL_DIR}/slurm_logs

mkdir -p "${AUDIT_DIR}" "${VALIDATION_DIR}" "${LOG_DIR}"

echo "host=$(hostname)"
echo "started_at=$(date -Is)"
python qa_data/localnewsqa_core/69_repair_final_explicit_single_cue.py

rm -f "${AUDIT_DIR}"/explicit_max_quality_failures.csv
rm -f "${AUDIT_DIR}"/explicit_max_quality_warnings.csv
rm -f "${AUDIT_DIR}"/explicit_max_quality_summary.json
rm -f "${VALIDATION_DIR}"/final_strict_explicit_*.csv
rm -f "${VALIDATION_DIR}"/final_strict_explicit_validation_summary.json
cp "${FINAL_FETCH}" "${AUDIT_DIR}/explicit_target_evidence_fetches.jsonl"

python qa_data/localnewsqa_core/48_audit_explicit_max_split.py \
  --input "${FINAL_JSONL}" \
  --ambiguous "${AMBIG}" \
  --outdir "${AUDIT_DIR}" \
  --fetch-missing \
  --retry-failed \
  --hydrate-wikipedia-api \
  --workers 32 \
  --timeout 25 \
  --wiki-batch-size 25 \
  --wiki-sleep 0.05

python qa_data/localnewsqa_core/67_validate_explicit_strict_defensible_1000_final.py \
  --input "${FINAL_JSONL}" \
  --ambiguous "${AMBIG}" \
  --outdir "${VALIDATION_DIR}" \
  --fetch-cache "${AUDIT_DIR}/explicit_target_evidence_fetches.jsonl"
echo "finished_at=$(date -Is)"
