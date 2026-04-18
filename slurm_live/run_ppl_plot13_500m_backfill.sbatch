#!/bin/bash
#SBATCH --job-name=ppl-plot13-500m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --time=0-12:00:00
#SBATCH --output=/scratch/amukher6/metacul/logs/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/amukher6/metacul/logs/slurm_logs/%x-%j.err

set -euo pipefail

mkdir -p /scratch/amukher6/metacul/logs/slurm_logs
source /home/amukher6/nanotron-env/bin/activate

cd /scratch/amukher6/metacul/src

BACKFILL_JSON=/scratch/amukher6/metacul/results/perplexity/eval_list_plot13_500m_backfill.json
BACKFILL_CSV=/scratch/amukher6/metacul/results/perplexity_eval_plot13_500m_backfill.csv
MAIN_CSV=/scratch/amukher6/metacul/results/perplexity_eval.csv

/home/amukher6/nanotron-env/bin/python 32_build_plot13_500m_eval_list.py
/home/amukher6/nanotron-env/bin/python 18_perplexity_eval.py \
  --eval-list "${BACKFILL_JSON}" \
  --output-csv "${BACKFILL_CSV}" \
  --local-files-only
/home/amukher6/nanotron-env/bin/python 29_merge_perplexity_csvs.py \
  --base "${MAIN_CSV}" \
  --incoming "${BACKFILL_CSV}" \
  --output "${MAIN_CSV}"
/home/amukher6/nanotron-env/bin/python - <<'PY'
import importlib.util
import matplotlib
matplotlib.use('Agg')
spec = importlib.util.spec_from_file_location('pplplot', '/scratch/amukher6/metacul/src/19_perplexity_plot.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.plot_token_efficiency_global_ppl()
print('rerendered plot13')
PY
