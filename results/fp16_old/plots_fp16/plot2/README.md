# Plot 2: Local vs Global on Local + Global Test Sets

- Source data: `/path/to/metacul/results/perplexity_eval.csv`
- Regions: Africa, America, Asia, Europe, and All (combined test set).
- Local bars: continent-specific models on their matching continent test sets.
- Global bars: combined models on each continent test set and on the combined test set.
- "All" local bars: aggregated from local model results on the combined test set.
  - Aggregation uses average log perplexity: `exp(mean(log(mean_ppl)))`.
  - CI bounds are aggregated similarly in log space using `ci_low` and `ci_high`.
- Metric: micro-average perplexity with 95% CI from the CSV.
- Y-axis lower bound fixed at 6 for comparability.
- Output files: `perplexity_local_vs_global_500m.pdf` and `perplexity_local_vs_global_1b.pdf`.
