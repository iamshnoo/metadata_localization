#!/bin/bash
#SBATCH --job-name=conv-500m-glob-int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --time=0-06:00:00
#SBATCH --output=/path/to/metacul/logs/slurm_logs/%x-%j.out
#SBATCH --error=/path/to/metacul/logs/slurm_logs/%x-%j.err

set -euo pipefail
MODEL_NAME="${1:?usage: sbatch $0 <model_name> <step>}"
STEP="${2:?usage: sbatch $0 <model_name> <step>}"
STEP_K=$((STEP / 1000))

source /path/to/nanotron-b200/bin/activate
mkdir -p /path/to/metacul/logs/slurm_logs

CKPT="/path/to/workspace/pretrain/logs/checkpoints/${MODEL_NAME}/${STEP}"
SAVE="/path/to/metacul/models/ablation_intermediates/metadata/${MODEL_NAME}_step${STEP_K}k"
SAVE_TMP="${SAVE}.tmp-${SLURM_JOB_ID}"
TOKENIZER_SRC="/path/to/metacul/models/combined_with_metadata_500m"
MASTER_PORT="$((20000 + (SLURM_JOB_ID % 20000)))"

cd /path/to/workspace/pretrain/src
rm -rf "$SAVE_TMP"

python -m torch.distributed.run --nproc_per_node=1 --master_port "$MASTER_PORT" convert_to_hf.py \
  --checkpoint_path "$CKPT" \
  --save_path "$SAVE_TMP" \
  --tokenizer_name "$TOKENIZER_SRC"

rm -rf "$SAVE"
mv "$SAVE_TMP" "$SAVE"
