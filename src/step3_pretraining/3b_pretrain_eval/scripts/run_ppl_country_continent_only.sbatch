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
#SBATCH --output=/scratch/amukher6/metacul/logs/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/amukher6/metacul/logs/slurm_logs/%x-%j.err

set -euo pipefail
mkdir -p /scratch/amukher6/metacul/logs/slurm_logs
source /home/amukher6/nanotron-env/bin/activate
cd /scratch/amukher6/metacul/src
/home/amukher6/nanotron-env/bin/python 28_build_country_continent_only_eval_list.py
/home/amukher6/nanotron-env/bin/python 18_perplexity_eval.py \
  --eval-list /scratch/amukher6/metacul/results/perplexity/eval_list_metadata_family_full.json \
  --output-csv /scratch/amukher6/metacul/results/perplexity_eval_metadata_family_full.csv \
  --local-files-only
