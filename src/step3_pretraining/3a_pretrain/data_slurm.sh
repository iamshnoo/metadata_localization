#!/bin/bash
#SBATCH --job-name=curlcym-tr
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --mem=80G
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --output=/scratch/$USER$/pretrain/nanotron/logs/slurm_logs/curlcym-tr.out
#SBATCH --error=/scratch/$USER$/pretrain/nanotron/logs/slurm_logs/curlcym-tr.err
#SBATCH --time=1-00:00:00

source ~/nanotron-env/bin/activate
ml load gnu12/12.3.0
ml load cuda/12.4.0
ml load git
set -x -e

echo "START TIME: $(date)"
secs_to_human() {
    echo "$(( ${1} / 3600 )):$(( (${1} / 60) % 60 )):$(( ${1} % 60 ))"
}
start=$(date +%s)
echo "$(date -d @${start} "+%Y-%m-%d %H:%M:%S"): ${SLURM_JOB_NAME} start id=${SLURM_JOB_ID}\n"

CMD="python tools/preprocess_data.py \
  --tokenizer-name-or-path meta-llama/Llama-3.2-1B \
  --output-folder /scratch/$USER$/pretrain/datasets/combined-only-url-country-with-metadata/train \
  --n-tasks 8 \
  local \
  --dataset /scratch/$USER$/metacul/training_data/meco_datasets/combined_only_url_country/with_metadata/train \
  --column text"

srun -u bash -c "$CMD"

echo "END TIME: $(date)"
end=$(date +%s)
elapsed=$((end - start))
echo "Total training time: $(secs_to_human $elapsed)"
