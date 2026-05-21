#!/bin/bash

# Usage:
#   sbatch --export=CONFIG_FILE=/path/to/workspace/pretrain/logs/configs/qwen100m_debug/combined_with_metadata_qwen100m.yaml \
#          /path/to/metacul/src/scripts/34_qwen100m_debug.slurm

#SBATCH --job-name=qwen100m-debug
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.40gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --wait-all-nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/logs/culture/out/qwen100m_debug.%A.out.txt
#SBATCH --error=/path/to/logs/culture/err/qwen100m_debug.%A.err.txt

set -euo pipefail

if [[ -z "${CONFIG_FILE:-}" ]]; then
  echo "CONFIG_FILE is required"
  exit 1
fi

RUN_NAME="${RUN_NAME:-$(basename "${CONFIG_FILE%.yaml}")}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

export PYTHON_BIN=/path/to/nanotron-b200/bin/python
export TORCHRUN_BIN=/path/to/nanotron-b200/bin/torchrun
export ACTIVATE_BIN=/path/to/nanotron-b200/bin/activate
export NANOTRON_DIR="${NANOTRON_DIR:-/path/to/workspace/pretrain/nanotron_full}"
export ENABLE_TIMERS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export STATS_SAMPLING_INTERVAL_IN_SEC=60
export MASTER_ADDR=$(hostname)
export MASTER_PORT="${MASTER_PORT:-6000}"
export TMPDIR=/tmp

echo "Run name: ${RUN_NAME}"
echo "Config: ${CONFIG_FILE}"
echo "Nanotron dir: ${NANOTRON_DIR}"
echo "GPUs/node: ${NPROC_PER_NODE}"

source "${ACTIVATE_BIN}"
ml load gnu12/12.3.0
ml load cuda/12.4.0
ml load git

cd "${NANOTRON_DIR}"
export PYTHONPATH="${NANOTRON_DIR}/src:${PYTHONPATH:-}"
"${PYTHON_BIN}" -c "import sys; print(sys.version)"
echo "Torchrun path: $(which torchrun)"
echo "Python path: $(which python)"
nvidia-smi

"${TORCHRUN_BIN}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --nnodes 1 \
  --rdzv_backend c10d \
  --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
  --max_restarts 0 \
  --tee 3 \
  run_train.py --config-file "${CONFIG_FILE}"
