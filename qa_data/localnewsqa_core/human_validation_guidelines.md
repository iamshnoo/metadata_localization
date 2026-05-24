# LocalNewsQA Human Validation Guidelines

This scaffold is designed for a targeted validation pass over `LocalNewsQA-Core-Ambiguous`.

## Why the ambiguous split

The ambiguous split is the benchmark component most directly tied to the paper's main claim. These items are only valid if:

1. the question is factually valid for the target locale,
2. the correct answer genuinely changes under the paired contrast locale, and
3. the wording does not trivially reveal the target country.

The explicit split is lower priority for human validation because it is closer to standard locale-grounded factual recall.

## Recommended sample size

- `20` items per country: `340` total
- `30` items per country: `510` total

Use at least two annotators per item.

## Annotation questions

For each sampled item, annotators should answer:

1. `judge_target_factuality`
   - `yes` if the target-locale answer is factually valid
   - `no` otherwise
2. `judge_locale_dependence`
   - `yes` if the paired contrast locale changes the correct answer
   - `no` otherwise
3. `judge_no_explicit_leakage`
   - `yes` if the wording does not explicitly reveal the target country
   - `no` otherwise

Annotators may add free-form notes in `annotator_notes`.

## Reporting

The central reported quantity should be agreement on `judge_locale_dependence`.

Report:

- percent agreement
- Cohen's `kappa` for two annotators, or Krippendorff's `alpha` if the setup expands
- marginal acceptance rate for each of the three judgments

## Sample builder

Generate a stratified CSV with:

```bash
python qa_data/localnewsqa_core/06_build_human_validation_sample.py \
  --dataset iamshnoo/LocalNewsQA \
  --split train \
  --per-country 20 \
  --output-csv qa_data/localnewsqa_core/runs/human_validation_ambiguous_340.csv
```
