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
#SBATCH --output=/scratch/amukher6/metacul/logs/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/amukher6/metacul/logs/slurm_logs/%x-%j.err

set -euo pipefail

mkdir -p /scratch/amukher6/metacul/logs/slurm_logs
set -a
source /scratch/amukher6/babel/llm-languages/.env >/dev/null 2>&1
set +a
source /home/amukher6/nanotron-env/bin/activate
cd /scratch/amukher6/metacul/src

/home/amukher6/nanotron-env/bin/python 15_sft_eval_grid.py \
  --size 3b \
  --name-suffix _best3b \
  --url-corruption-rates 0.0 \
  --results-dir /scratch/amukher6/metacul/results/downstream_3b_best \
  --summary-csv /scratch/amukher6/metacul/results/qa_metacul_eval_summary_3b_best.csv
