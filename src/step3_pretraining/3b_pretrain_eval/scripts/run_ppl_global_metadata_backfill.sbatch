#!/bin/bash
#SBATCH --job-name=ppl-global-meta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --time=2-00:00:00
#SBATCH --output=/path/to/metacul/logs/slurm_logs/%x-%j.out
#SBATCH --error=/path/to/metacul/logs/slurm_logs/%x-%j.err

set -euo pipefail

mkdir -p /path/to/metacul/logs/slurm_logs
source /path/to/nanotron-env/bin/activate

cd /path/to/metacul/src

BACKFILL_JSON=/path/to/metacul/results/perplexity/eval_list_global_metadata_family_backfill.json
BACKFILL_CSV=/path/to/metacul/results/perplexity_eval_global_metadata_backfill.csv
MAIN_CSV=/path/to/metacul/results/perplexity_eval.csv

/path/to/nanotron-env/bin/python 31_build_global_metadata_family_eval_list.py
/path/to/nanotron-env/bin/python 18_perplexity_eval.py \
  --eval-list "${BACKFILL_JSON}" \
  --output-csv "${BACKFILL_CSV}" \
  --local-files-only
/path/to/nanotron-env/bin/python 29_merge_perplexity_csvs.py \
  --base "${MAIN_CSV}" \
  --incoming "${BACKFILL_CSV}" \
  --output "${MAIN_CSV}"
/path/to/nanotron-env/bin/python 19_perplexity_plot.py
