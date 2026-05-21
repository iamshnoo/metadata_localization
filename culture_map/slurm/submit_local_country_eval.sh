#!/bin/bash
set -euo pipefail

SLURM_SCRIPT=/path/to/culture-map/slurm/run_local_country_eval_single.slurm
LOG_DIR=/path/to/logs/slurm_logs
mkdir -p "$LOG_DIR"

timestamp=$(date +%Y%m%d_%H%M%S)
manifest="$LOG_DIR/cmap_local_country_eval_${timestamp}.tsv"
printf "job_id\tvariant\n" > "$manifest"

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
  printf "%s\t%s\n" "$job_id" "$variant" >> "$manifest"
  echo "$job_id  $variant"
done

echo "Manifest: $manifest"
