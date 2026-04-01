# Plot 6: Metadata Ablations (1B)

- Source data: `/scratch/amukher6/metacul/results/perplexity_eval.csv`
- Models: metadata ablation models (URL / URL+Continent / URL+Country) plus combined with/without models and their step checkpoints.
- Checkpoints: step2k, step4k, step8k, and final (10k).
- Panels (3):
  - Panel 1: models on their own metadata-ablation test sets ([URL], [URL][Continent], [URL][Country]).
    - Combined models use the mean across the three metadata‑ablation test sets at each checkpoint.
  - Panel 2: models on combined with-metadata test set ([ALL]).
  - Panel 3: models on combined without-metadata test set (ALL).
- Missing checkpoints are left as gaps and will render once results are available.
- Output file: `perplexity_metadata_ablations_1b.pdf`.
