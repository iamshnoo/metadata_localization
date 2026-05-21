#!/bin/bash
#SBATCH --job-name=merge-sft-3b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --time=0-08:00:00
#SBATCH --output=/path/to/metacul/logs/slurm_logs/%x-%j.out
#SBATCH --error=/path/to/metacul/logs/slurm_logs/%x-%j.err

set -euo pipefail

VARIANT="${1:?usage: sbatch $0 <with|without>}"

mkdir -p /path/to/metacul/logs/slurm_logs
source /path/to/nanotron-env/bin/activate
cd /path/to/metacul/src

if [[ "$VARIANT" == "with" ]]; then
  /path/to/nanotron-env/bin/python 13_merge_lora.py --size 3b --metadata
elif [[ "$VARIANT" == "without" ]]; then
  /path/to/nanotron-env/bin/python 13_merge_lora.py --size 3b
else
  echo "Unknown variant: $VARIANT" >&2
  exit 1
fi
