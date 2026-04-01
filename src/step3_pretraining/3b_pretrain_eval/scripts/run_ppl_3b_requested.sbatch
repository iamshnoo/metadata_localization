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
#SBATCH --output=/scratch/amukher6/metacul/logs/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/amukher6/metacul/logs/slurm_logs/%x-%j.err

set -euo pipefail

mkdir -p /scratch/amukher6/metacul/logs/slurm_logs
source /home/amukher6/nanotron-env/bin/activate

cd /scratch/amukher6/metacul/src
/home/amukher6/nanotron-env/bin/python 17_eval_list.py
/home/amukher6/nanotron-env/bin/python 27_build_3b_ppl_eval_list.py
/home/amukher6/nanotron-env/bin/python 18_perplexity_eval.py \
  --eval-list /scratch/amukher6/metacul/results/perplexity/eval_list_3b_requested.json \
  --output-csv /scratch/amukher6/metacul/results/perplexity_eval.csv \
  --local-files-only
