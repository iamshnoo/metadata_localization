#!/bin/bash
#SBATCH --job-name=hf-collect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=contrib
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/amukher6/metacul/logs/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/amukher6/metacul/logs/slurm_logs/%x-%j.err

set -euo pipefail

EXTRA_ARGS="${*:-}"

set -a
source /scratch/amukher6/babel/llm-languages/.env >/dev/null 2>&1
set +a
source /home/amukher6/nanotron-env/bin/activate

python3 /scratch/amukher6/metacul/src/30_sync_hf_collection_and_model_cards.py ${EXTRA_ARGS}
