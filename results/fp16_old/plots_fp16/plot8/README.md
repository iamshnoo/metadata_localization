# Plot 8: Apples-to-Apples QA Accuracy

- Source data: `/path/to/metacul/results/qa_metacul_eval.csv`
- Filter: only rows where `answered_by_all == 1`.
- Metric: accuracy = correct / (correct + incorrect).
- Models: Custom vs LLaMA-3, each with/without metadata.
- Styling: metadata condition encoded by color; model family encoded by hatch.
- Left panel: per continent and model, compute accuracy per `base_url` then plot mean and 95% CI across the 10 URLs.
- Right panel: per model, compute a continent-weighted accuracy per `base_url` (weights = answered-by-all question counts per continent) then plot mean and 95% CI across the 10 URLs.
- Output file: `accuracy_apples_to_apples_answered_by_all.pdf`.
