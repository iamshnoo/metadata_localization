#!/bin/bash
#SBATCH --job-name=merge-sft-1b-f2500
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --time=0-08:00:00
#SBATCH --output=/path/to/metacul/logs/slurm_logs/%x-%j.out
#SBATCH --error=/path/to/metacul/logs/slurm_logs/%x-%j.err

set -euo pipefail

VARIANT="${1:?usage: sbatch $0 <with|without>}"
PYTHON=/path/to/nanotron-b200/bin/python
NAME_SUFFIX=${NAME_SUFFIX:-_final2500}
mkdir -p /path/to/metacul/logs/slurm_logs

export HF_HOME=/path/to/cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

cd /path/to/metacul/src

if [[ "$VARIANT" == "with" ]]; then
  "$PYTHON" 13_merge_lora.py --size 1b --metadata --name-suffix "$NAME_SUFFIX"
elif [[ "$VARIANT" == "without" ]]; then
  "$PYTHON" 13_merge_lora.py --size 1b --name-suffix "$NAME_SUFFIX"
else
  echo "Unknown variant: $VARIANT" >&2
  exit 1
fi
