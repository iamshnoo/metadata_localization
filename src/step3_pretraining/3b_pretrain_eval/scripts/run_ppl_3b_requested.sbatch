#!/bin/bash
#SBATCH --job-name=ppl-3b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --time=3-00:00:00
#SBATCH --output=/path/to/metacul/logs/slurm_logs/%x-%j.out
#SBATCH --error=/path/to/metacul/logs/slurm_logs/%x-%j.err

set -euo pipefail

mkdir -p /path/to/metacul/logs/slurm_logs
source /path/to/nanotron-env/bin/activate

cd /path/to/metacul/src
/path/to/nanotron-env/bin/python 17_eval_list.py
/path/to/nanotron-env/bin/python 27_build_3b_ppl_eval_list.py
/path/to/nanotron-env/bin/python 18_perplexity_eval.py \
  --eval-list /path/to/metacul/results/perplexity/eval_list_3b_requested.json \
  --output-csv /path/to/metacul/results/perplexity_eval.csv \
  --local-files-only
