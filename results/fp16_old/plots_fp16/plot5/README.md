# Plot 5: Cross-Continent Asymmetry (Local Models)

- Source data: `/path/to/metacul/results/perplexity_eval.csv`
- Models/tests: local models and local test sets only, with metadata ([Local] | [Test]).
- Asymmetry(i, j) = P(i→j) − P(j→i).
- P(i→j): perplexity when model trained on i is evaluated on j.

How to read the heatmap:
- Positive value at (i, j): j→i generalizes better than i→j (since lower perplexity is better).
- Negative value at (i, j): i→j generalizes better than j→i.
- Near zero: roughly symmetric cross‑generalization.
- Diagonal is ~0 by construction.

- Two figures: one for 500m and one for 1b.
- Output files:
  - `perplexity_asymmetry_500m.pdf`
  - `perplexity_asymmetry_1b.pdf`
