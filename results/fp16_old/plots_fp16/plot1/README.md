# Plot 1: Continent Metadata Effect

- Source data: `/scratch/amukher6/metacul/results/perplexity_eval.csv`
- Selection: continent models evaluated on their own continent test sets.
- Bars: four metadata combinations (model/test) using the labels shown in the legend.
- Metric: micro-average perplexity from `mean_ppl` with 95% CI (`ci_low`, `ci_high`).
- Y-axis lower bound fixed at 6 for comparability.
- Output files: `perplexity_continent_metadata_effect_500m.pdf` and `perplexity_continent_metadata_effect_1b.pdf`.
