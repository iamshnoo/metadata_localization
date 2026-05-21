#!/usr/bin/env bash
# Rebuild LocalNewsQA Table 11, dependent figures, and PDFs after the
# Table-12-roster gold LocalNewsQA jobs finish.
set -euo pipefail

ROOT=/path/to/metacul
PYTHON=${PYTHON:-/path/to/nanotron-b200/bin/python}

cd "$ROOT"
mkdir -p results/localnewsqa_gold_20260516/plots

"$PYTHON" src/72_localnewsqa_gold_model_gain_tables.py --strict --seed 41 --bootstrap 2000

"$PYTHON" src/70_plot_downstream_gain_pareto_frontier.py \
  --external-gains results/appendix_model_gain_tables_20260505/external_model_gains_long.csv \
  --local-gains results/localnewsqa_gold_20260516/appendix_model_gain_tables/localnewsqa_model_gains_long.csv \
  --results-dir results/localnewsqa_gold_20260516/plots \
  --basename downstream_ambiguous_gain_pareto_frontier \
  --local-metric localnewsqa_ambiguous \
  --label-mode essential

cp \
  results/localnewsqa_gold_20260516/plots/downstream_ambiguous_gain_pareto_frontier.pdf \
  latex/figs/main/9_downstream_ambiguous_gain_pareto_frontier.pdf

(
  cd latex
  latexmk -pdf -interaction=nonstopmode -halt-on-error arr_may26.tex
)

"$PYTHON" - <<'PY'
from pathlib import Path

root = Path("/path/to/metacul")
src = root / "latex/arr_may26.tex"
out = root / "latex/overleaf_bundle/arr_may.tex"
text = src.read_text(encoding="utf-8")
replacements = {
    r"\input{../results/localnewsqa_gold_20260516/appendix_model_gain_tables/localnewsqa_model_gains_tabular.tex}": (
        root
        / "results/localnewsqa_gold_20260516/appendix_model_gain_tables/localnewsqa_model_gains_tabular.tex"
    ).read_text(encoding="utf-8"),
    r"\input{../results/appendix_model_gain_tables_20260505/external_model_gains_tabular.tex}": (
        root / "results/appendix_model_gain_tables_20260505/external_model_gains_tabular.tex"
    ).read_text(encoding="utf-8"),
}
for needle, repl in replacements.items():
    if needle not in text:
        raise SystemExit(f"missing input marker: {needle}")
    text = text.replace(needle, repl)
out.write_text(text, encoding="utf-8")
PY

rsync -a latex/figs/ latex/overleaf_bundle/figs/
cp latex/anthology.bib latex/custom.bib latex/acl.sty latex/acl_natbib.bst latex/overleaf_bundle/

(
  cd latex/overleaf_bundle
  latexmk -pdf -interaction=nonstopmode -halt-on-error arr_may.tex
)

rg -n "Overfull|Underfull|LaTeX Warning|Package .* Warning|Rerun|undefined" \
  latex/arr_may26.log latex/overleaf_bundle/arr_may.log || true
