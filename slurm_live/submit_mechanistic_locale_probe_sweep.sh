#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SLURM_SCRIPT="$SCRIPT_DIR/mechanistic_locale_probe_single.slurm"
LOG_DIR=/path/to/logs/slurm_logs
mkdir -p "$LOG_DIR"

timestamp=$(date +%Y%m%d_%H%M%S)
manifest="$LOG_DIR/locale_probe_sweep_${timestamp}.tsv"
printf "variant\tprompt_kind\tmodel_path\tjob_id\n" > "$manifest"

submit() {
  local variant="$1"
  local prompt_kind="$2"
  local model_path="$3"
  local job_id
  job_id=$(sbatch --parsable --export=ALL,VARIANT_LABEL="$variant",PROMPT_KIND="$prompt_kind",MODEL_PATH="$model_path" "$SLURM_SCRIPT")
  printf "%s\t%s\t%s\t%s\n" "$variant" "$prompt_kind" "$model_path" "$job_id" >> "$manifest"
  echo "$variant -> $job_id"
}

for step in 2k 4k 8k final; do
  if [[ "$step" == "final" ]]; then
    submit "url_final" "url" "/path/to/metacul/models/ablations/metadata/combined_only_url_with_metadata_1b"
    submit "country_final" "country" "/path/to/metacul/models/combined_only_country_with_metadata_1b"
    submit "continent_final" "continent" "/path/to/metacul/models/combined_only_continent_with_metadata_1b"
    submit "none_final" "none" "/path/to/metacul/models/combined_without_metadata_1b"
  else
    submit "url_step${step}" "url" "/path/to/metacul/models/ablation_intermediates/metadata/combined_only_url_with_metadata_1b_step${step}"
    submit "country_step${step}" "country" "/path/to/metacul/models/ablation_intermediates/metadata/combined_only_country_with_metadata_1b_step${step}"
    submit "continent_step${step}" "continent" "/path/to/metacul/models/ablation_intermediates/metadata/combined_only_continent_with_metadata_1b_step${step}"
    submit "none_step${step}" "none" "/path/to/metacul/models/ablation_intermediates/metadata/combined_without_metadata_1b_step${step}"
  fi
done

echo "Manifest: $manifest"
