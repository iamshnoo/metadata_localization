#!/bin/bash

# Usage:
#   sbatch --export=CONFIG_FILE=/scratch/amukher6/metacul/qwen_transfer_configs/combined_with_metadata_qwen1b.yaml \
#          /scratch/amukher6/metacul/src/scripts/34_qwen1b_transfer.slurm
#
# Optional overrides:
#   --export=CONFIG_FILE=...,RUN_NAME=qwen1b_with_meta,NNODES=1,NPROC_PER_NODE=4

#SBATCH --job-name=qwen1b-transfer
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:4
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=250000M
#SBATCH --wait-all-nodes=1
#SBATCH --time=5-00:00:00
#SBATCH --output=/scratch/amukher6/logs/culture/out/qwen1b_transfer.%A.out.txt
#SBATCH --error=/scratch/amukher6/logs/culture/err/qwen1b_transfer.%A.err.txt

set -euo pipefail

if [[ -z "${CONFIG_FILE:-}" ]]; then
  echo "CONFIG_FILE is required"
  exit 1
fi

RUN_NAME="${RUN_NAME:-$(basename "${CONFIG_FILE%.yaml}")}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

export PYTHON_BIN=/scratch/amukher6/nanotron-b200/bin/python
export TORCHRUN_BIN=/scratch/amukher6/nanotron-b200/bin/torchrun
export ACTIVATE_BIN=/scratch/amukher6/nanotron-b200/bin/activate
export ENABLE_TIMERS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export STATS_SAMPLING_INTERVAL_IN_SEC=60
export HOSTNAMES=$(scontrol show hostnames "${SLURM_JOB_NODELIST}")
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT="${MASTER_PORT:-6000}"
export COUNT_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | wc -l)
export TMPDIR=/tmp

echo "Run name: ${RUN_NAME}"
echo "Config: ${CONFIG_FILE}"
echo "Nodes: ${NNODES}"
echo "GPUs/node: ${NPROC_PER_NODE}"

source "${ACTIVATE_BIN}"
ml load gnu12/12.3.0
ml load cuda/12.4.0
ml load git

cd /scratch/amukher6/pretrain/nanotron
"${PYTHON_BIN}" -c "import sys; print(sys.version)"
echo "NVCC version: $(nvcc --version | tail -n 1)"
echo "Torchrun path: $(which torchrun)"
echo "Python path: $(which python)"
nvidia-smi

LAUNCHER="${TORCHRUN_BIN} \
  --nproc_per_node ${NPROC_PER_NODE} \
  --nnodes ${COUNT_NODE} \
  --rdzv_backend c10d \
  --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
  --max_restarts 0 \
  --tee 3"

srun -u bash -c "${LAUNCHER} run_train.py --config-file ${CONFIG_FILE}"
