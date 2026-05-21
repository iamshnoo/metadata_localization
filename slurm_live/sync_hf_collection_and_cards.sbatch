#!/bin/bash
#SBATCH --job-name=hf-collect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=contrib
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/metacul/logs/slurm_logs/%x-%j.out
#SBATCH --error=/path/to/metacul/logs/slurm_logs/%x-%j.err

set -euo pipefail

EXTRA_ARGS="${*:-}"

set -a
# Set required tokens, e.g. HF_TOKEN, in the environment before running.
set +a
source /path/to/nanotron-env/bin/activate

python3 /path/to/metacul/src/30_sync_hf_collection_and_model_cards.py ${EXTRA_ARGS}
