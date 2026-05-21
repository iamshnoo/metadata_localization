#!/bin/bash
#SBATCH --job-name=qwen-arch-ppl10k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --time=1-00:00:00
#SBATCH --output=/path/to/metacul/logs/slurm_logs/%x-%j.out
#SBATCH --error=/path/to/metacul/logs/slurm_logs/%x-%j.err

set -euo pipefail

ROOT=/path/to/metacul
NANOTRON=/path/to/workspace/pretrain/nanotron_full
CONVERT_DIR=/path/to/workspace/pretrain/src
CKPT_ROOT=/path/to/workspace/pretrain/logs/checkpoints/qwen1b_pretrain_full
OUT_ROOT="${ROOT}/models/qwen_arch_generalization"
TOKENIZER_SRC="${ROOT}/models/combined_with_metadata_3b"
EVAL_LIST="${ROOT}/results/perplexity/eval_list_qwen_arch_10k.json"
OUT_CSV="${ROOT}/results/perplexity_eval_qwen_arch_10k.csv"
PER_SAMPLE_DIR="${ROOT}/results/per_sample_losses_qwen_arch_10k"
WITH_CKPT="${CKPT_ROOT}/combined_with_metadata_qwen1b/10000"
WITHOUT_CKPT="${CKPT_ROOT}/combined_without_metadata_qwen1b/10000"
WITH_HF="${OUT_ROOT}/combined_with_metadata_qwen1b_step10k_hf"
WITHOUT_HF="${OUT_ROOT}/combined_without_metadata_qwen1b_step10k_hf"
MASTER_PORT="$((22000 + (SLURM_JOB_ID % 20000)))"

mkdir -p "${ROOT}/logs/slurm_logs" "${OUT_ROOT}"
source /path/to/nanotron-b200/bin/activate
export PYTHONPATH="${NANOTRON}/examples/llama:${NANOTRON}/src:${PYTHONPATH:-}"

convert_one() {
  local ckpt="$1"
  local save="$2"
  local tmp="${save}.tmp-${SLURM_JOB_ID}"

  rm -rf "${tmp}"
  cd "${CONVERT_DIR}"
  python -m torch.distributed.run --nproc_per_node=1 --master_port "${MASTER_PORT}" convert_to_hf.py \
    --checkpoint_path "${ckpt}" \
    --save_path "${tmp}" \
    --tokenizer_name "${TOKENIZER_SRC}" \
    --config_cls Qwen2Config
  rm -rf "${save}"
  mv "${tmp}" "${save}"
}

if [[ ! -d "${WITH_HF}" ]]; then
  convert_one "${WITH_CKPT}" "${WITH_HF}"
fi

if [[ ! -d "${WITHOUT_HF}" ]]; then
  convert_one "${WITHOUT_CKPT}" "${WITHOUT_HF}"
fi

cd "${ROOT}/src"
python 18_perplexity_eval.py \
  --eval-list "${EVAL_LIST}" \
  --output-csv "${OUT_CSV}" \
  --per-sample-dir "${PER_SAMPLE_DIR}" \
  --tokenizer-path "${TOKENIZER_SRC}" \
  --local-files-only \
  --max-samples 1000 \
  --bootstrap-iters 1000
