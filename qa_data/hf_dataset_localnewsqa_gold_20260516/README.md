---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-00000-of-00001.parquet
---

# LocalNewsQA

LocalNewsQA is an English-news benchmark for localized knowledge disambiguation. This release contains the final gold split used in the ARR May 2026 submission experiments.

## Release

- Version: `localnewsqa_gold_20260516`
- Total rows: 18,700
- Explicit rows: 17,000 (1,000 per country)
- Ambiguous rows: 1,700 (100 per country)
- Countries: 17
- Topic labels: 16, grouped into the 8 topic families used in the paper

The explicit split names the locale in the question. The ambiguous split omits the locale from the question and includes target/contrast metadata with different supported answers. Rows include cached evidence titles/URLs/excerpts when available and split-specific validation status fields.

## Columns

Core fields: `id`, `question`, `options`, `correct_answer`, `distractors`, `country`, `continent`, `topic`, `year`, `split_type`, `ambiguity_flag`.

Locale fields: `target_country`, `contrast_country`, `target_answer`, `contrast_answer`.

Evidence fields: `evidence_hint`, `target_evidence_title`, `target_evidence_url`, `target_evidence_excerpt`, `contrast_evidence_title`, `contrast_evidence_url`, `contrast_evidence_excerpt`, `target_evidence_cached`, `contrast_evidence_cached`.

Validation fields: `relation_support_status`, `curation_status`, `source_row_id`, `release_version`.
