#!/bin/bash
#SBATCH --job-name=sft-3b-best-with
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

export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/amukher6/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/amukher6/.cache/huggingface/transformers
export HF_DATASETS_CACHE=/scratch/amukher6/.cache/huggingface/datasets

cd /scratch/amukher6/metacul/src

srun /home/amukher6/nanotron-env/bin/python 12_sft.py \
  --size 3b \
  --metadata \
  --profile best3b \
  --name-suffix _best3b
