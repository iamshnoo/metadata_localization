#!/bin/bash
#SBATCH --job-name=ppl-meta-family
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --time=1-00:00:00
#SBATCH --output=/path/to/metacul/logs/slurm_logs/%x-%j.out
#SBATCH --error=/path/to/metacul/logs/slurm_logs/%x-%j.err

set -euo pipefail
mkdir -p /path/to/metacul/logs/slurm_logs
source /path/to/nanotron-env/bin/activate
cd /path/to/metacul/src
/path/to/nanotron-env/bin/python 28_build_country_continent_only_eval_list.py
/path/to/nanotron-env/bin/python 18_perplexity_eval.py \
  --eval-list /path/to/metacul/results/perplexity/eval_list_metadata_family_full.json \
  --output-csv /path/to/metacul/results/perplexity_eval_metadata_family_full.csv \
  --local-files-only
