# Plot 3: Global Model Scaling (500m vs 1b)

- Source data: `/path/to/metacul/results/perplexity_eval.csv`
- Models: global (combined) models at 500m and 1b.
- Test sets: Africa, America, Asia, Europe, and All (combined).
- Single figure with both metadata conditions; two dumbbells per region.
- Plot type: vertical dumbbell (500m vs 1b) with delta annotations (Δ = 1b − 500m).
- Metric: micro-average perplexity with 95% CI from the CSV.
- X-axis lower bound fixed at 6 for comparability.
- Output file: `perplexity_scaling_global_delta.pdf`.
