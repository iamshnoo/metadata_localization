#!/bin/bash
#SBATCH --job-name=sft-3b-without
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

export PYTHONUNBUFFERED=1
export HF_HOME=/path/to/cache/huggingface
export TRANSFORMERS_CACHE=/path/to/cache/huggingface/transformers
export HF_DATASETS_CACHE=/path/to/cache/huggingface/datasets

cd /path/to/metacul/src

srun /path/to/nanotron-env/bin/python 12_sft.py --size 3b
