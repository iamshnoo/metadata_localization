# Plot 7: Leave-One-Out Ablations (1B)

- Source data: `/path/to/metacul/results/perplexity_eval.csv`
- Models:
  - Combined models and their step checkpoints.
  - Leave‑one‑out models (no Africa/America/Asia/Europe) and their step checkpoints.
- Two figures: one for with‑metadata and one for without‑metadata.
- Two panels per figure:
  - Left: evaluation on the left‑out continent.
    - Combined line is mean across left‑out continent test sets.
    - Leave‑one‑out lines are per‑continent (NoAfrica/NoAmerica/NoAsia/NoEurope).
  - Right: evaluation on combined test set.
    - Combined line is direct on the combined test set.
    - Leave‑one‑out lines are per‑continent (NoAfrica/NoAmerica/NoAsia/NoEurope).
- Checkpoints: step2k, step4k, step8k, final (10k).
- Output files:
  - `leave_one_out_with_metadata.pdf`
  - `leave_one_out_without_metadata.pdf`
