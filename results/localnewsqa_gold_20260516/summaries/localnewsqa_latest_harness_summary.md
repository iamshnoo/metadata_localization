# LocalNewsQA Gold Harness Summary

This summary documents the LocalNewsQA gold evaluation artifacts used by the current paper build.

## Dataset

- Gold data: `qa_data/localnewsqa_core/final_gold_20260516/localnewsqa_gold_explicit17000_ambiguous1700.jsonl`
- Size: 18,700 rows total, with 17,000 explicit rows and 1,700 ambiguous rows.
- Balance: 1,000 explicit rows and 100 ambiguous rows for each of 17 countries.
- Main answer-order seed: 41.
- Bootstrap: question-level intervals in the generated appendix tables.

## Paper Table/Figure Sources

- LocalNewsQA model-gain table: `results/localnewsqa_gold_20260516/appendix_model_gain_tables/localnewsqa_model_gains_long.csv`
- Full per-cell protocol manifest: `results/localnewsqa_gold_20260516/appendix_model_gain_tables/localnewsqa_protocol_manifest.md`
- Figure 2 target/switch source CSVs:
  - `results/localnewsqa_gold_20260516/plots/plot_8_pretrained_target_split_seed41_bootstrap.csv`
  - `results/localnewsqa_gold_20260516/summaries/localnewsqa_locale_switch/summary.csv`
- Figure 6b LocalNewsQA source: `results/localnewsqa_gold_20260516/appendix_model_gain_tables/localnewsqa_model_gains_long.csv`
- Figure 6b external source: `results/appendix_model_gain_tables_20260505/external_model_gains_long.csv`

## MAPLE Harnesses Used In The Latest LocalNewsQA Tables

| model | source directory | metadata prompt | QA prompt | answer cue | option labels | scoring | calibration | BOS |
|---|---|---|---|---|---|---|---|---|
| MAPLE 1B Base | `results/localnewsqa_gold_20260516/pretrained_target/1b_codeg_labels_question_final/seed_41` | `code_grounded` / `none` | `question` | `final_answer_colon` | kept | `option_text_avg`, alpha=1.25 | none | no |
| MAPLE 3B Base | `results/localnewsqa_gold_20260516/pretrained_target/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41` | `name_grounded` / `none` | `question_answer` | `final_answer_colon` | omitted | `option_text_avg`, alpha=0.25 | `question_masked`, beta=0.5 | yes |
| MAPLE 1B Chat | `results/localnewsqa_gold_20260516/sft_target/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41` | `name_plain` / `none` | `question_answer` | `country_final_answer_colon` | omitted | `option_text_avg`, alpha=2.5 | `question_masked`, beta=0.7 | yes |
| MAPLE 3B Chat | `results/localnewsqa_gold_20260516/sft_target/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41` | `name_grounded` / `none` | `question_answer` | `final_answer_colon` | omitted | `option_text_avg`, alpha=0.25 | `question_masked`, beta=0.5 | yes |

For each MAPLE row, the plus side is `(T+, I+)` and the minus side is `(T-, I-)`; target prompts use target locale metadata and contrast prompts use contrast locale metadata.

## Latest Core MAPLE Results

| model | overall gain | explicit gain | ambiguous gain | exact-pair gain | margin-switch gain |
|---|---:|---:|---:|---:|---:|
| MAPLE 1B Base | +1.39 | +0.86 | +6.71 | +1.88 | +8.24 |
| MAPLE 3B Base | +2.79 | +2.15 | +9.24 | +8.76 | +25.12 |
| MAPLE 1B Chat | +1.20 | +0.88 | +4.41 | +2.00 | +6.53 |
| MAPLE 3B Chat | +2.20 | +1.41 | +10.12 | +5.12 | +13.65 |

The old 35,874-row LocalNewsQA retained pool is not the reported benchmark. Current paper tables and main LocalNewsQA figures use the 18,700-row gold split above.
