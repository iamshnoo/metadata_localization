#!/bin/bash
#SBATCH --job-name=sft-1b-f2500-with
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --time=1-00:00:00
#SBATCH --output=/path/to/metacul/logs/slurm_logs/%x-%j.out
#SBATCH --error=/path/to/metacul/logs/slurm_logs/%x-%j.err

set -euo pipefail

PYTHON=/path/to/nanotron-b200/bin/python
NAME_SUFFIX=${NAME_SUFFIX:-_final2500}
MAX_STEPS=${MAX_STEPS:-2500}
SAVE_STEPS=${SAVE_STEPS:-500}
EVAL_STEPS=${EVAL_STEPS:-500}
mkdir -p /path/to/metacul/logs/slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME=/path/to/cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

cd /path/to/metacul/src

srun "$PYTHON" 12_sft.py \
  --size 1b \
  --metadata \
  --name-suffix "$NAME_SUFFIX" \
  --max-steps "$MAX_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --eval-steps "$EVAL_STEPS"
