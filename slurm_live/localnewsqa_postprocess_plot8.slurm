#!/bin/bash
#SBATCH --job-name=lnqa-post-plot8
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=24GB
#SBATCH --time=0-04:00:00
#SBATCH --output=/scratch/amukher6/logs/slurm_logs/lnqa_post_plot8.%j.out
#SBATCH --error=/scratch/amukher6/logs/slurm_logs/lnqa_post_plot8.%j.err

set -euo pipefail

PYTHON=/home/amukher6/nanotron-env/bin/python
RESULTS_DIR=/scratch/amukher6/metacul/results/downstream_localnewsqa
MATRIX_CSV=/scratch/amukher6/metacul/results/qa_metacul_eval.csv
PLOT_SRC=/scratch/amukher6/metacul/results/plots/plot8/accuracy_apples_to_apples_answered_by_all.pdf
PLOT_DST=/scratch/amukher6/metacul/latex/figs/main/8_accuracy_apples_to_apples_answered_by_all.pdf

"$PYTHON" /scratch/amukher6/metacul/src/16_sft_analyse.py \
  --results-dir "$RESULTS_DIR" \
  --output-csv "$MATRIX_CSV"

"$PYTHON" - <<'PY'
import runpy
ns = runpy.run_path('/scratch/amukher6/metacul/src/19_perplexity_plot.py', run_name='plotmod')
ns['plot_sft_accuracy_apples_to_apples']()
PY

cp "$PLOT_SRC" "$PLOT_DST"
echo "Postprocess complete: updated $MATRIX_CSV and $PLOT_DST"

