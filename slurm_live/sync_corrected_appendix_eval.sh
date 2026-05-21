#!/bin/bash
set -euo pipefail

ROOT=/path/to/metacul
PYTHON=/path/to/nanotron-env/bin/python

LOCALNEWSQA_PRETRAINED_ROOT="$ROOT/results/downstream_localnewsqa_pretrained_raw_code_grounded_multiseed"
TABLE8_1B_ROOT="$ROOT/results/external_benchmarks_pretrained_raw_code_grounded/1b"
TABLE8_3B_ROOT="$ROOT/results/external_benchmarks_pretrained_raw_code_grounded/3b"
ADVERSARIAL_ROOT="$ROOT/results/downstream_localnewsqa_pretrained_adversarial_raw_code_grounded"
LATEX_APPENDIX_DIR="$ROOT/latex/figs/appendix"

mkdir -p "$LATEX_APPENDIX_DIR"

echo "[1/6] Aggregating corrected LocalNewsQA multiseed MAPLE results"
"$PYTHON" "$ROOT/src/35_localnewsqa_multiseed_summary.py" \
  --pretrained-root "$LOCALNEWSQA_PRETRAINED_ROOT"

echo "[2/6] Replotting Figure 9 from corrected MAPLE outputs"
"$PYTHON" "$ROOT/src/34_plot_localnewsqa_figure9.py"

echo "[3/6] Aggregating corrected Table 8 MAPLE external results"
"$PYTHON" "$ROOT/src/36_external_benchmarks_multiseed_summary.py" \
  --ours-1b-root "$TABLE8_1B_ROOT" \
  --ours-3b-root "$TABLE8_3B_ROOT"

echo "[4/6] Recomputing external bootstrap delta plot from corrected MAPLE outputs"
"$PYTHON" "$ROOT/src/27_external_negligible_analysis.py" \
  --ours-1b-root "$TABLE8_1B_ROOT" \
  --ours-3b-root "$TABLE8_3B_ROOT"
cp "$ROOT/results/plots/plot14/external_negligible_delta_bootstrap.pdf" \
   "$LATEX_APPENDIX_DIR/19_external_negligible_delta_bootstrap.pdf"
echo "[ok] Copied corrected external bootstrap plot into latex appendix figs"

echo "[5/6] Aggregating corrected adversarial LocalNewsQA results"
"$PYTHON" "$ROOT/src/44_pretrained_adversarial_summary.py" \
  --input-dir "$ADVERSARIAL_ROOT"

echo "[6/6] Rendering corrected adversarial appendix plots"
"$PYTHON" "$ROOT/src/45_render_adversarial_plots.py"

echo "[done] Corrected appendix MAPLE eval artifacts are synced."
