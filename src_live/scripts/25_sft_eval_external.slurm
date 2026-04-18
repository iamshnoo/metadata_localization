#!/bin/bash
#SBATCH -p contrib-gpuq
#SBATCH -q cs_dept
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --mem=40G
#SBATCH --time=0-6:00:00
#SBATCH -J sft_ext_eval
#SBATCH -o /scratch/amukher6/metacul/results/external_benchmarks/logs/sft_ext_eval_%j.out
#SBATCH -e /scratch/amukher6/metacul/results/external_benchmarks/logs/sft_ext_eval_%j.err

set -euo pipefail

mkdir -p /scratch/amukher6/metacul/results/external_benchmarks/logs

if [[ -f /scratch/amukher6/metacul/src/scripts/.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /scratch/amukher6/metacul/src/scripts/.env
  set +a
fi

echo "[INFO] Host: $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[INFO] HF_TOKEN is set."
else
  echo "[INFO] HF_TOKEN is NOT set."
fi
echo "[INFO] Start: $(date)"

/home/amukher6/nanotron-env/bin/python /scratch/amukher6/metacul/src/25_sft_eval_external.py "$@"

echo "[INFO] End: $(date)"
