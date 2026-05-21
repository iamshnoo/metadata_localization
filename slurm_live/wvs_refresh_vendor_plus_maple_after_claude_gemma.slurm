#!/bin/bash
#SBATCH --job-name=wvs-refresh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=contrib
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/metacul/logs/slurm_logs/%x-%j.out
#SBATCH --error=/path/to/metacul/logs/slurm_logs/%x-%j.err

set -euo pipefail

: "${UPSTREAM_JOBS:?UPSTREAM_JOBS must be set}"

PYTHON_BIN=${PYTHON_BIN:-/path/to/nanotron-b200/bin/python}
CULTURE_MAP_ROOT=/path/to/culture-map
METACUL_ROOT=/path/to/metacul
DATA_DIR=${DATA_DIR:-$CULTURE_MAP_ROOT/data/paper_osf}
PLOT_SCRIPT=${PLOT_SCRIPT:-$METACUL_ROOT/src/48_plot_wvs_vendor_plus_maple.py}

export PYTHONPATH=$CULTURE_MAP_ROOT/src${PYTHONPATH:+:$PYTHONPATH}

echo "Triggered at $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Upstream jobs: $UPSTREAM_JOBS"
sacct -j "$UPSTREAM_JOBS" --format=JobID,JobName%20,State,Elapsed,Start,End,ExitCode -P || true

echo
echo "Merging Gemma shard outputs into canonical directory..."
"$PYTHON_BIN" - <<'PY'
from pathlib import Path

from culture_map.provider_country_runner import merge_country_model_shards

data_dir = Path("/path/to/culture-map/data/paper_osf")
canonical_dir = Path("/path/to/culture-map/outputs/provider_country_eval_all_human_split/gemma-4-31b-it")
shard_dirs = [
    canonical_dir,
    Path("/path/to/culture-map/outputs/provider_country_eval_all_human_split/gemma-4-31b-it_shard2"),
    Path("/path/to/culture-map/outputs/provider_country_eval_all_human_split/gemma-4-31b-it_shard3"),
    Path("/path/to/culture-map/outputs/provider_country_eval_all_human_split/gemma-4-31b-it_shard4"),
    Path("/path/to/culture-map/outputs/provider_country_eval_all_human_split/gemma-4-31b-it_key1_resume"),
    Path("/path/to/culture-map/outputs/provider_country_eval_all_human_split/gemma-4-31b-it_key2_resume"),
]

outputs = merge_country_model_shards(
    data_dir=data_dir,
    shard_output_dirs=shard_dirs,
    output_dir=canonical_dir,
    model_name="gemma-4-31b-it",
    label="Gemma 4 31B",
    country_scope="all_human",
    metadata_prompt_style="name_typical",
)
print(canonical_dir / "all_model_mean_projection.csv")
print(outputs["mean_projection"].to_string(index=False))
PY

echo
echo "Rebuilding WVS vendor plus MAPLE plots..."
"$PYTHON_BIN" "$PLOT_SCRIPT"

echo
echo "Updated PDFs:"
echo "/path/to/metacul/latex/figs/appendix/21_wvs_vendor_plus_maple_nometa_official_axes.pdf"
echo "/path/to/metacul/latex/figs/appendix/21_wvs_vendor_plus_maple_official_axes.pdf"
echo "Completed at $(date '+%Y-%m-%d %H:%M:%S %Z')"
