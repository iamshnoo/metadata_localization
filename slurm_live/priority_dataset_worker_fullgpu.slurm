#!/bin/bash
#SBATCH --job-name=priority-dataset-worker
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96GB
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/path/to/logs/slurm_logs/priority_dataset_worker.%j.out
#SBATCH --error=/path/to/logs/slurm_logs/priority_dataset_worker.%j.err

set -euo pipefail

: "${CHUNK_SCRIPT:?CHUNK_SCRIPT must point to a generated worker chunk script}"

echo "Running chunk: $CHUNK_SCRIPT"
bash "$CHUNK_SCRIPT"
