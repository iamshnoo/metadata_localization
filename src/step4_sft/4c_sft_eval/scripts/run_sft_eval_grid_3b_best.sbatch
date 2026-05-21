#!/bin/bash
#SBATCH --job-name=sft-eval-3b-best
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --time=1-12:00:00
#SBATCH --output=/path/to/metacul/logs/slurm_logs/%x-%j.out
#SBATCH --error=/path/to/metacul/logs/slurm_logs/%x-%j.err

set -euo pipefail

mkdir -p /path/to/metacul/logs/slurm_logs
set -a
# Set required tokens, e.g. HF_TOKEN, in the environment before running.
set +a
source /path/to/nanotron-env/bin/activate
cd /path/to/metacul/src

/path/to/nanotron-env/bin/python 15_sft_eval_grid.py \
  --size 3b \
  --name-suffix _best3b \
  --url-corruption-rates 0.0 \
  --results-dir /path/to/metacul/results/downstream_3b_best \
  --summary-csv /path/to/metacul/results/qa_metacul_eval_summary_3b_best.csv
