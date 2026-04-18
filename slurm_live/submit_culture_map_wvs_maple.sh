#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SLURM_SCRIPT="$SCRIPT_DIR/culture_map_wvs_country_eval_single.slurm"
LOG_DIR=/scratch/amukher6/logs/slurm_logs
mkdir -p "$LOG_DIR"

timestamp=$(date +%Y%m%d_%H%M%S)
manifest="$LOG_DIR/culture_map_wvs_maple_${timestamp}.tsv"
printf "variant\tjob_id\n" > "$manifest"

variants=(
  maple_1b_tplus_eplus
  maple_1b_tplus_eminus
  maple_1b_tminus_eplus
  maple_1b_tminus_eminus
  maple_3b_tplus_eplus
  maple_3b_tplus_eminus
  maple_3b_tminus_eplus
  maple_3b_tminus_eminus
)

for variant in "${variants[@]}"; do
  job_id=$(sbatch --parsable --export=ALL,VARIANT="$variant" "$SLURM_SCRIPT")
  printf "%s\t%s\n" "$variant" "$job_id" >> "$manifest"
  echo "$variant -> $job_id"
done

echo "Manifest: $manifest"
