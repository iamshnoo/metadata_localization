# Appendix Table Protocol Manifest

This file documents every model/dataset cell in the two large appendix model-gain tables.
It is generated from the table CSV artifacts and the source JSONLs used by each cell.

## Reproduction Commands

1. `python src/66_appendix_model_gain_tables.py --seed 41 --bootstrap 2000 --bootstrap-seed 20260505`
2. `python src/67_appendix_table_protocol_manifest.py [--table-out-dir DIR] [--local-only]`

## Fixed Illustrative Example

Every protocol rendering below uses the same toy item so prompt-format changes are easy to compare.

```text
COUNTRY: Canada
COUNTRY_CODE: ca
CONTINENT: America
QUESTION: Which institution is the national public broadcaster for this locale?
OPTIONS:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio
```

## Cells

### 1. LocalNewsQA appendix model-gain table | MAPLE 1B Base | LocalNewsQA Overall

- model name: `MAPLE 1B Base`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta +1.39, CI [+0.8503, +1.93], n=18700
- positive source protocol key: `figure9_1b`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=code_grounded; qa_prompt_style=question; answer_cue_style=final_answer_colon; omit_option_labels=False; exact_option_text_instruction=True; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=1.25; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=final_answer_colon; omit_option_labels=False; exact_option_text_instruction=True; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=1.25; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/pretrained_target/1b_codeg_labels_question_final/seed_41/localnewsqa_eval_target_with_metadata_custom_frozenfig9_1b_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/pretrained_target/1b_codeg_labels_question_final/seed_41/localnewsqa_eval_target_without_metadata_custom_frozenfig9_1b_tminus_eminus_seed41_c0.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
COUNTRY_NAME: Canada
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Use the locale metadata above as grounding. This question is about Canada.

Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the exact text of the correct option.

Final answer:
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the exact text of the correct option.

Final answer:
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 2. LocalNewsQA appendix model-gain table | MAPLE 1B Base | LocalNewsQA Ambiguous

- model name: `MAPLE 1B Base`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +6.71, CI [+5.06, +8.35], n=1700
- positive source protocol key: `figure9_1b`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=code_grounded; qa_prompt_style=question; answer_cue_style=final_answer_colon; omit_option_labels=False; exact_option_text_instruction=True; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=1.25; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=final_answer_colon; omit_option_labels=False; exact_option_text_instruction=True; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=1.25; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/pretrained_target/1b_codeg_labels_question_final/seed_41/localnewsqa_eval_target_with_metadata_custom_frozenfig9_1b_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/pretrained_target/1b_codeg_labels_question_final/seed_41/localnewsqa_eval_target_without_metadata_custom_frozenfig9_1b_tminus_eminus_seed41_c0.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
COUNTRY_NAME: Canada
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Use the locale metadata above as grounding. This question is about Canada.

Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the exact text of the correct option.

Final answer:
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the exact text of the correct option.

Final answer:
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 3. LocalNewsQA appendix model-gain table | MAPLE 1B Base | LocalNewsQA Explicit

- model name: `MAPLE 1B Base`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta +0.8588, CI [+0.2881, +1.39], n=17000
- positive source protocol key: `figure9_1b`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=code_grounded; qa_prompt_style=question; answer_cue_style=final_answer_colon; omit_option_labels=False; exact_option_text_instruction=True; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=1.25; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=final_answer_colon; omit_option_labels=False; exact_option_text_instruction=True; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=1.25; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/pretrained_target/1b_codeg_labels_question_final/seed_41/localnewsqa_eval_target_with_metadata_custom_frozenfig9_1b_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/pretrained_target/1b_codeg_labels_question_final/seed_41/localnewsqa_eval_target_without_metadata_custom_frozenfig9_1b_tminus_eminus_seed41_c0.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
COUNTRY_NAME: Canada
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Use the locale metadata above as grounding. This question is about Canada.

Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the exact text of the correct option.

Final answer:
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the exact text of the correct option.

Final answer:
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 4. LocalNewsQA appendix model-gain table | MAPLE 1B Base | LocalNewsQA Exact pair

- model name: `MAPLE 1B Base`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +1.88, CI [+1.24, +2.53], n=1700
- positive source protocol key: `figure9_1b`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=code_grounded; qa_prompt_style=question; answer_cue_style=final_answer_colon; omit_option_labels=False; exact_option_text_instruction=True; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=1.25; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=final_answer_colon; omit_option_labels=False; exact_option_text_instruction=True; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=1.25; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/pretrained_target/1b_codeg_labels_question_final/seed_41/localnewsqa_eval_target_with_metadata_custom_frozenfig9_1b_tplus_eplus_seed41_c0.jsonl`
  - `results/localnewsqa_gold_20260516/pretrained_contrast/1b_codeg_labels_question_final/seed_41/localnewsqa_eval_contrast_with_metadata_custom_frozenfig9contrast_1b_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/pretrained_target/1b_codeg_labels_question_final/seed_41/localnewsqa_eval_target_without_metadata_custom_frozenfig9_1b_tminus_eminus_seed41_c0.jsonl`
  - `results/localnewsqa_gold_20260516/pretrained_contrast/1b_codeg_labels_question_final/seed_41/localnewsqa_eval_contrast_without_metadata_custom_frozenfig9contrast_1b_tminus_eminus_seed41_c0.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
COUNTRY_NAME: Canada
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Use the locale metadata above as grounding. This question is about Canada.

Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the exact text of the correct option.

Final answer:
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the exact text of the correct option.

Final answer:
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 5. LocalNewsQA appendix model-gain table | MAPLE 1B Base | LocalNewsQA Margin switch

- model name: `MAPLE 1B Base`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +8.24, CI [+6.82, +9.59], n=1700
- positive source protocol key: `figure9_1b`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=code_grounded; qa_prompt_style=question; answer_cue_style=final_answer_colon; omit_option_labels=False; exact_option_text_instruction=True; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=1.25; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=final_answer_colon; omit_option_labels=False; exact_option_text_instruction=True; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=1.25; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/pretrained_target/1b_codeg_labels_question_final/seed_41/localnewsqa_eval_target_with_metadata_custom_frozenfig9_1b_tplus_eplus_seed41_c0.jsonl`
  - `results/localnewsqa_gold_20260516/pretrained_contrast/1b_codeg_labels_question_final/seed_41/localnewsqa_eval_contrast_with_metadata_custom_frozenfig9contrast_1b_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/pretrained_target/1b_codeg_labels_question_final/seed_41/localnewsqa_eval_target_without_metadata_custom_frozenfig9_1b_tminus_eminus_seed41_c0.jsonl`
  - `results/localnewsqa_gold_20260516/pretrained_contrast/1b_codeg_labels_question_final/seed_41/localnewsqa_eval_contrast_without_metadata_custom_frozenfig9contrast_1b_tminus_eminus_seed41_c0.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
COUNTRY_NAME: Canada
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Use the locale metadata above as grounding. This question is about Canada.

Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the exact text of the correct option.

Final answer:
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the exact text of the correct option.

Final answer:
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 6. LocalNewsQA appendix model-gain table | MAPLE 1B Chat | LocalNewsQA Overall

- model name: `MAPLE 1B Chat`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta +1.20, CI [+0.5775, +1.88], n=18700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_plain; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=2.5; null_calibration_mode=question_masked; null_calibration_beta=0.7; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=2.5; null_calibration_mode=question_masked; null_calibration_beta=0.7; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/sft_target/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_sftgold_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/sft_target/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_sftgold_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tminus_eminus_seed41_c0.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: Canada
COUNTRY_CODE: ca
CONTINENT: America

TITLE: Facts about Canada

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

For Canada, final answer:
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

Final answer:
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 7. LocalNewsQA appendix model-gain table | MAPLE 1B Chat | LocalNewsQA Ambiguous

- model name: `MAPLE 1B Chat`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +4.41, CI [+2.41, +6.47], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_plain; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=2.5; null_calibration_mode=question_masked; null_calibration_beta=0.7; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=2.5; null_calibration_mode=question_masked; null_calibration_beta=0.7; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/sft_target/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_sftgold_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/sft_target/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_sftgold_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tminus_eminus_seed41_c0.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: Canada
COUNTRY_CODE: ca
CONTINENT: America

TITLE: Facts about Canada

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

For Canada, final answer:
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

Final answer:
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 8. LocalNewsQA appendix model-gain table | MAPLE 1B Chat | LocalNewsQA Explicit

- model name: `MAPLE 1B Chat`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta +0.8765, CI [+0.1824, +1.56], n=17000
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_plain; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=2.5; null_calibration_mode=question_masked; null_calibration_beta=0.7; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=2.5; null_calibration_mode=question_masked; null_calibration_beta=0.7; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/sft_target/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_sftgold_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/sft_target/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_sftgold_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tminus_eminus_seed41_c0.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: Canada
COUNTRY_CODE: ca
CONTINENT: America

TITLE: Facts about Canada

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

For Canada, final answer:
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

Final answer:
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 9. LocalNewsQA appendix model-gain table | MAPLE 1B Chat | LocalNewsQA Exact pair

- model name: `MAPLE 1B Chat`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +2.00, CI [+1.41, +2.71], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_plain; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=2.5; null_calibration_mode=question_masked; null_calibration_beta=0.7; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=2.5; null_calibration_mode=question_masked; null_calibration_beta=0.7; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/sft_target/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_sftgold_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tplus_eplus_seed41_c0.jsonl`
  - `results/localnewsqa_gold_20260516/sft_contrast/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41/localnewsqa_eval_contrast_with_metadata_custom_sftgoldcontrast_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/sft_target/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_sftgold_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tminus_eminus_seed41_c0.jsonl`
  - `results/localnewsqa_gold_20260516/sft_contrast/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41/localnewsqa_eval_contrast_without_metadata_custom_sftgoldcontrast_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tminus_eminus_seed41_c0.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: Canada
COUNTRY_CODE: ca
CONTINENT: America

TITLE: Facts about Canada

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

For Canada, final answer:
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

Final answer:
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 10. LocalNewsQA appendix model-gain table | MAPLE 1B Chat | LocalNewsQA Margin switch

- model name: `MAPLE 1B Chat`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +6.53, CI [+5.41, +7.71], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_plain; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=2.5; null_calibration_mode=question_masked; null_calibration_beta=0.7; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=2.5; null_calibration_mode=question_masked; null_calibration_beta=0.7; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/sft_target/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_sftgold_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tplus_eplus_seed41_c0.jsonl`
  - `results/localnewsqa_gold_20260516/sft_contrast/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41/localnewsqa_eval_contrast_with_metadata_custom_sftgoldcontrast_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/sft_target/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_sftgold_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tminus_eminus_seed41_c0.jsonl`
  - `results/localnewsqa_gold_20260516/sft_contrast/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41/localnewsqa_eval_contrast_without_metadata_custom_sftgoldcontrast_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tminus_eminus_seed41_c0.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: Canada
COUNTRY_CODE: ca
CONTINENT: America

TITLE: Facts about Canada

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

For Canada, final answer:
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

Final answer:
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 11. LocalNewsQA appendix model-gain table | MAPLE 3B Base | LocalNewsQA Overall

- model name: `MAPLE 3B Base`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta +2.79, CI [+1.96, +3.65], n=18700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_grounded; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/pretrained_target/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_frozenfig9_3b_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/pretrained_target/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_frozenfig9_3b_tminus_eminus_seed41_c0.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: Canada
COUNTRY_CODE: ca
CONTINENT: America

TITLE: Facts about Canada

CONTENT:
Use the locale metadata above as grounding. Answer for Canada, not a different country.

Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

Final answer:
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

Final answer:
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 12. LocalNewsQA appendix model-gain table | MAPLE 3B Base | LocalNewsQA Ambiguous

- model name: `MAPLE 3B Base`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +9.24, CI [+6.47, +11.77], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_grounded; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/pretrained_target/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_frozenfig9_3b_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/pretrained_target/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_frozenfig9_3b_tminus_eminus_seed41_c0.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: Canada
COUNTRY_CODE: ca
CONTINENT: America

TITLE: Facts about Canada

CONTENT:
Use the locale metadata above as grounding. Answer for Canada, not a different country.

Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

Final answer:
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

Final answer:
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 13. LocalNewsQA appendix model-gain table | MAPLE 3B Base | LocalNewsQA Explicit

- model name: `MAPLE 3B Base`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta +2.15, CI [+1.25, +3.01], n=17000
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_grounded; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/pretrained_target/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_frozenfig9_3b_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/pretrained_target/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_frozenfig9_3b_tminus_eminus_seed41_c0.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: Canada
COUNTRY_CODE: ca
CONTINENT: America

TITLE: Facts about Canada

CONTENT:
Use the locale metadata above as grounding. Answer for Canada, not a different country.

Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

Final answer:
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

Final answer:
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 14. LocalNewsQA appendix model-gain table | MAPLE 3B Base | LocalNewsQA Exact pair

- model name: `MAPLE 3B Base`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +8.76, CI [+7.41, +10.12], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_grounded; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/pretrained_target/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_frozenfig9_3b_tplus_eplus_seed41_c0.jsonl`
  - `results/localnewsqa_gold_20260516/pretrained_contrast/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_contrast_with_metadata_custom_frozenfig9contrast_3b_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/pretrained_target/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_frozenfig9_3b_tminus_eminus_seed41_c0.jsonl`
  - `results/localnewsqa_gold_20260516/pretrained_contrast/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_contrast_without_metadata_custom_frozenfig9contrast_3b_tminus_eminus_seed41_c0.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: Canada
COUNTRY_CODE: ca
CONTINENT: America

TITLE: Facts about Canada

CONTENT:
Use the locale metadata above as grounding. Answer for Canada, not a different country.

Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

Final answer:
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

Final answer:
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 15. LocalNewsQA appendix model-gain table | MAPLE 3B Base | LocalNewsQA Margin switch

- model name: `MAPLE 3B Base`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +25.12, CI [+23.12, +27.18], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_grounded; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/pretrained_target/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_frozenfig9_3b_tplus_eplus_seed41_c0.jsonl`
  - `results/localnewsqa_gold_20260516/pretrained_contrast/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_contrast_with_metadata_custom_frozenfig9contrast_3b_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/pretrained_target/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_frozenfig9_3b_tminus_eminus_seed41_c0.jsonl`
  - `results/localnewsqa_gold_20260516/pretrained_contrast/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_contrast_without_metadata_custom_frozenfig9contrast_3b_tminus_eminus_seed41_c0.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: Canada
COUNTRY_CODE: ca
CONTINENT: America

TITLE: Facts about Canada

CONTENT:
Use the locale metadata above as grounding. Answer for Canada, not a different country.

Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

Final answer:
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

Final answer:
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 16. LocalNewsQA appendix model-gain table | MAPLE 3B Chat | LocalNewsQA Overall

- model name: `MAPLE 3B Chat`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta +2.20, CI [+1.39, +3.04], n=18700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_grounded; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/sft_target/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_sftgold_3b_best3b_chat_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/sft_target/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_sftgold_3b_best3b_chat_tminus_eminus_seed41_c0.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: Canada
COUNTRY_CODE: ca
CONTINENT: America

TITLE: Facts about Canada

CONTENT:
Use the locale metadata above as grounding. Answer for Canada, not a different country.

Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

Final answer:
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

Final answer:
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 17. LocalNewsQA appendix model-gain table | MAPLE 3B Chat | LocalNewsQA Ambiguous

- model name: `MAPLE 3B Chat`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +10.12, CI [+7.47, +12.82], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_grounded; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/sft_target/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_sftgold_3b_best3b_chat_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/sft_target/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_sftgold_3b_best3b_chat_tminus_eminus_seed41_c0.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: Canada
COUNTRY_CODE: ca
CONTINENT: America

TITLE: Facts about Canada

CONTENT:
Use the locale metadata above as grounding. Answer for Canada, not a different country.

Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

Final answer:
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

Final answer:
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 18. LocalNewsQA appendix model-gain table | MAPLE 3B Chat | LocalNewsQA Explicit

- model name: `MAPLE 3B Chat`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta +1.41, CI [+0.5294, +2.24], n=17000
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_grounded; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/sft_target/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_sftgold_3b_best3b_chat_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/sft_target/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_sftgold_3b_best3b_chat_tminus_eminus_seed41_c0.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: Canada
COUNTRY_CODE: ca
CONTINENT: America

TITLE: Facts about Canada

CONTENT:
Use the locale metadata above as grounding. Answer for Canada, not a different country.

Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

Final answer:
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

Final answer:
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 19. LocalNewsQA appendix model-gain table | MAPLE 3B Chat | LocalNewsQA Exact pair

- model name: `MAPLE 3B Chat`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +5.12, CI [+4.12, +6.18], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_grounded; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/sft_target/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_sftgold_3b_best3b_chat_tplus_eplus_seed41_c0.jsonl`
  - `results/localnewsqa_gold_20260516/sft_contrast/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_contrast_with_metadata_custom_sftgoldcontrast_3b_best3b_chat_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/sft_target/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_sftgold_3b_best3b_chat_tminus_eminus_seed41_c0.jsonl`
  - `results/localnewsqa_gold_20260516/sft_contrast/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_contrast_without_metadata_custom_sftgoldcontrast_3b_best3b_chat_tminus_eminus_seed41_c0.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: Canada
COUNTRY_CODE: ca
CONTINENT: America

TITLE: Facts about Canada

CONTENT:
Use the locale metadata above as grounding. Answer for Canada, not a different country.

Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

Final answer:
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

Final answer:
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 20. LocalNewsQA appendix model-gain table | MAPLE 3B Chat | LocalNewsQA Margin switch

- model name: `MAPLE 3B Chat`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +13.65, CI [+12.06, +15.24], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_grounded; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/sft_target/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_sftgold_3b_best3b_chat_tplus_eplus_seed41_c0.jsonl`
  - `results/localnewsqa_gold_20260516/sft_contrast/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_contrast_with_metadata_custom_sftgoldcontrast_3b_best3b_chat_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/sft_target/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_sftgold_3b_best3b_chat_tminus_eminus_seed41_c0.jsonl`
  - `results/localnewsqa_gold_20260516/sft_contrast/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_contrast_without_metadata_custom_sftgoldcontrast_3b_best3b_chat_tminus_eminus_seed41_c0.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: Canada
COUNTRY_CODE: ca
CONTINENT: America

TITLE: Facts about Canada

CONTENT:
Use the locale metadata above as grounding. Answer for Canada, not a different country.

Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

Final answer:
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
- Canadian Broadcasting Corporation
- British Broadcasting Corporation
- Australian Broadcasting Corporation
- National Public Radio

Final answer:
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 21. LocalNewsQA appendix model-gain table | LLaMA-3.2-1B Inst. | LocalNewsQA Overall

- model name: `LLaMA-3.2-1B Inst.`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta -2.25, CI [-2.64, -1.82], n=18700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_llama32_1b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_llama32_1b_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 22. LocalNewsQA appendix model-gain table | LLaMA-3.2-1B Inst. | LocalNewsQA Ambiguous

- model name: `LLaMA-3.2-1B Inst.`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +0.4118, CI [-1.06, +1.88], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_llama32_1b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_llama32_1b_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 23. LocalNewsQA appendix model-gain table | LLaMA-3.2-1B Inst. | LocalNewsQA Explicit

- model name: `LLaMA-3.2-1B Inst.`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta -2.52, CI [-2.96, -2.09], n=17000
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_llama32_1b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_llama32_1b_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 24. LocalNewsQA appendix model-gain table | LLaMA-3.2-1B Inst. | LocalNewsQA Exact pair

- model name: `LLaMA-3.2-1B Inst.`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +3.06, CI [+2.24, +3.94], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_llama32_1b_with_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_llama32_1b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_llama32_1b_without_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_llama32_1b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 25. LocalNewsQA appendix model-gain table | LLaMA-3.2-1B Inst. | LocalNewsQA Margin switch

- model name: `LLaMA-3.2-1B Inst.`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +9.18, CI [+7.88, +10.59], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_llama32_1b_with_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_llama32_1b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_llama32_1b_without_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_llama32_1b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 26. LocalNewsQA appendix model-gain table | LLaMA-3.2-3B Inst. | LocalNewsQA Overall

- model name: `LLaMA-3.2-3B Inst.`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta -8.01, CI [-8.56, -7.55], n=18700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_llama32_3b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_llama32_3b_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 27. LocalNewsQA appendix model-gain table | LLaMA-3.2-3B Inst. | LocalNewsQA Ambiguous

- model name: `LLaMA-3.2-3B Inst.`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta -2.65, CI [-4.41, -0.8809], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_llama32_3b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_llama32_3b_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 28. LocalNewsQA appendix model-gain table | LLaMA-3.2-3B Inst. | LocalNewsQA Explicit

- model name: `LLaMA-3.2-3B Inst.`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta -8.55, CI [-9.07, -8.02], n=17000
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_llama32_3b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_llama32_3b_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 29. LocalNewsQA appendix model-gain table | LLaMA-3.2-3B Inst. | LocalNewsQA Exact pair

- model name: `LLaMA-3.2-3B Inst.`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +3.65, CI [+2.76, +4.59], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_llama32_3b_with_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_llama32_3b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_llama32_3b_without_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_llama32_3b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 30. LocalNewsQA appendix model-gain table | LLaMA-3.2-3B Inst. | LocalNewsQA Margin switch

- model name: `LLaMA-3.2-3B Inst.`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +10.59, CI [+9.12, +12.00], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_llama32_3b_with_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_llama32_3b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_llama32_3b_without_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_llama32_3b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 31. LocalNewsQA appendix model-gain table | Qwen2.5-0.5B Inst. | LocalNewsQA Overall

- model name: `Qwen2.5-0.5B Inst.`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta -0.8128, CI [-1.17, -0.4813], n=18700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_0p5b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_0p5b_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 32. LocalNewsQA appendix model-gain table | Qwen2.5-0.5B Inst. | LocalNewsQA Ambiguous

- model name: `Qwen2.5-0.5B Inst.`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +1.53, CI [+0.2941, +2.76], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_0p5b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_0p5b_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 33. LocalNewsQA appendix model-gain table | Qwen2.5-0.5B Inst. | LocalNewsQA Explicit

- model name: `Qwen2.5-0.5B Inst.`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta -1.05, CI [-1.39, -0.7000], n=17000
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_0p5b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_0p5b_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 34. LocalNewsQA appendix model-gain table | Qwen2.5-0.5B Inst. | LocalNewsQA Exact pair

- model name: `Qwen2.5-0.5B Inst.`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +2.24, CI [+1.59, +2.94], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_0p5b_with_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_qwen25_0p5b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_0p5b_without_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_qwen25_0p5b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 35. LocalNewsQA appendix model-gain table | Qwen2.5-0.5B Inst. | LocalNewsQA Margin switch

- model name: `Qwen2.5-0.5B Inst.`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +5.76, CI [+4.76, +6.88], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_0p5b_with_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_qwen25_0p5b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_0p5b_without_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_qwen25_0p5b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 36. LocalNewsQA appendix model-gain table | Qwen2.5-1.5B Inst. | LocalNewsQA Overall

- model name: `Qwen2.5-1.5B Inst.`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta -1.83, CI [-2.28, -1.38], n=18700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_1p5b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_1p5b_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 37. LocalNewsQA appendix model-gain table | Qwen2.5-1.5B Inst. | LocalNewsQA Ambiguous

- model name: `Qwen2.5-1.5B Inst.`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +8.53, CI [+6.71, +10.29], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_1p5b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_1p5b_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 38. LocalNewsQA appendix model-gain table | Qwen2.5-1.5B Inst. | LocalNewsQA Explicit

- model name: `Qwen2.5-1.5B Inst.`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta -2.86, CI [-3.31, -2.41], n=17000
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_1p5b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_1p5b_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 39. LocalNewsQA appendix model-gain table | Qwen2.5-1.5B Inst. | LocalNewsQA Exact pair

- model name: `Qwen2.5-1.5B Inst.`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +7.71, CI [+6.47, +9.00], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_1p5b_with_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_qwen25_1p5b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_1p5b_without_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_qwen25_1p5b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 40. LocalNewsQA appendix model-gain table | Qwen2.5-1.5B Inst. | LocalNewsQA Margin switch

- model name: `Qwen2.5-1.5B Inst.`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +20.41, CI [+18.53, +22.18], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_1p5b_with_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_qwen25_1p5b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_1p5b_without_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_qwen25_1p5b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 41. LocalNewsQA appendix model-gain table | Qwen2.5-3B Inst. | LocalNewsQA Overall

- model name: `Qwen2.5-3B Inst.`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta +0.6150, CI [+0.2299, +0.9841], n=18700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_3b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_3b_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 42. LocalNewsQA appendix model-gain table | Qwen2.5-3B Inst. | LocalNewsQA Ambiguous

- model name: `Qwen2.5-3B Inst.`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +8.71, CI [+6.94, +10.41], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_3b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_3b_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 43. LocalNewsQA appendix model-gain table | Qwen2.5-3B Inst. | LocalNewsQA Explicit

- model name: `Qwen2.5-3B Inst.`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta -0.1941, CI [-0.5590, +0.1706], n=17000
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_3b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_3b_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 44. LocalNewsQA appendix model-gain table | Qwen2.5-3B Inst. | LocalNewsQA Exact pair

- model name: `Qwen2.5-3B Inst.`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +8.24, CI [+7.00, +9.59], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_3b_with_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_qwen25_3b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_3b_without_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_qwen25_3b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 45. LocalNewsQA appendix model-gain table | Qwen2.5-3B Inst. | LocalNewsQA Margin switch

- model name: `Qwen2.5-3B Inst.`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +25.24, CI [+23.12, +27.29], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_3b_with_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_qwen25_3b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen25_3b_without_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_qwen25_3b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 46. LocalNewsQA appendix model-gain table | Qwen3.5-0.8B | LocalNewsQA Overall

- model name: `Qwen3.5-0.8B`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta -1.07, CI [-1.43, -0.6951], n=18700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_0p8b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_0p8b_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 47. LocalNewsQA appendix model-gain table | Qwen3.5-0.8B | LocalNewsQA Ambiguous

- model name: `Qwen3.5-0.8B`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +3.76, CI [+2.47, +5.06], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_0p8b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_0p8b_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 48. LocalNewsQA appendix model-gain table | Qwen3.5-0.8B | LocalNewsQA Explicit

- model name: `Qwen3.5-0.8B`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta -1.56, CI [-1.95, -1.16], n=17000
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_0p8b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_0p8b_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 49. LocalNewsQA appendix model-gain table | Qwen3.5-0.8B | LocalNewsQA Exact pair

- model name: `Qwen3.5-0.8B`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +2.24, CI [+1.59, +3.00], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_0p8b_with_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_qwen35_0p8b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_0p8b_without_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_qwen35_0p8b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 50. LocalNewsQA appendix model-gain table | Qwen3.5-0.8B | LocalNewsQA Margin switch

- model name: `Qwen3.5-0.8B`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +8.29, CI [+7.00, +9.59], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_0p8b_with_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_qwen35_0p8b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_0p8b_without_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_qwen35_0p8b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 51. LocalNewsQA appendix model-gain table | Qwen3.5-2B | LocalNewsQA Overall

- model name: `Qwen3.5-2B`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta +2.65, CI [+2.24, +3.08], n=18700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_2b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_2b_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 52. LocalNewsQA appendix model-gain table | Qwen3.5-2B | LocalNewsQA Ambiguous

- model name: `Qwen3.5-2B`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +11.24, CI [+9.47, +13.12], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_2b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_2b_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 53. LocalNewsQA appendix model-gain table | Qwen3.5-2B | LocalNewsQA Explicit

- model name: `Qwen3.5-2B`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta +1.79, CI [+1.36, +2.19], n=17000
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_2b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_2b_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 54. LocalNewsQA appendix model-gain table | Qwen3.5-2B | LocalNewsQA Exact pair

- model name: `Qwen3.5-2B`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +8.53, CI [+7.24, +9.88], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_2b_with_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_qwen35_2b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_2b_without_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_qwen35_2b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 55. LocalNewsQA appendix model-gain table | Qwen3.5-2B | LocalNewsQA Margin switch

- model name: `Qwen3.5-2B`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +20.41, CI [+18.59, +22.29], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_2b_with_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_qwen35_2b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_2b_without_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_qwen35_2b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 56. LocalNewsQA appendix model-gain table | Qwen3.5-4B | LocalNewsQA Overall

- model name: `Qwen3.5-4B`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta +0.1711, CI [-0.0643, +0.4332], n=18700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_4b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_4b_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 57. LocalNewsQA appendix model-gain table | Qwen3.5-4B | LocalNewsQA Ambiguous

- model name: `Qwen3.5-4B`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +1.29, CI [+0.4706, +2.18], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_4b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_4b_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 58. LocalNewsQA appendix model-gain table | Qwen3.5-4B | LocalNewsQA Explicit

- model name: `Qwen3.5-4B`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta +0.0588, CI [-0.2059, +0.3176], n=17000
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_4b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_4b_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 59. LocalNewsQA appendix model-gain table | Qwen3.5-4B | LocalNewsQA Exact pair

- model name: `Qwen3.5-4B`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +0.8235, CI [+0.4118, +1.24], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_4b_with_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_qwen35_4b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_4b_without_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_qwen35_4b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 60. LocalNewsQA appendix model-gain table | Qwen3.5-4B | LocalNewsQA Margin switch

- model name: `Qwen3.5-4B`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +3.12, CI [+2.35, +4.00], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_4b_with_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_qwen35_4b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_4b_without_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_qwen35_4b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 61. LocalNewsQA appendix model-gain table | Qwen3.5-9B | LocalNewsQA Overall

- model name: `Qwen3.5-9B`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta -0.0107, CI [-0.2620, +0.2299], n=18700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_9b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_9b_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 62. LocalNewsQA appendix model-gain table | Qwen3.5-9B | LocalNewsQA Ambiguous

- model name: `Qwen3.5-9B`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +1.18, CI [+0.2941, +2.06], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_9b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_9b_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 63. LocalNewsQA appendix model-gain table | Qwen3.5-9B | LocalNewsQA Explicit

- model name: `Qwen3.5-9B`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta -0.1294, CI [-0.3706, +0.1118], n=17000
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_9b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_9b_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 64. LocalNewsQA appendix model-gain table | Qwen3.5-9B | LocalNewsQA Exact pair

- model name: `Qwen3.5-9B`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +0.7059, CI [+0.3529, +1.12], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_9b_with_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_qwen35_9b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_9b_without_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_qwen35_9b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 65. LocalNewsQA appendix model-gain table | Qwen3.5-9B | LocalNewsQA Margin switch

- model name: `Qwen3.5-9B`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +2.24, CI [+1.59, +2.94], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_9b_with_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_qwen35_9b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_qwen35_9b_without_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_qwen35_9b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 66. LocalNewsQA appendix model-gain table | Gemma-4-E2B-it | LocalNewsQA Overall

- model name: `Gemma-4-E2B-it`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta +1.03, CI [+0.6791, +1.38], n=18700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_gemma4_e2b_it_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_gemma4_e2b_it_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 67. LocalNewsQA appendix model-gain table | Gemma-4-E2B-it | LocalNewsQA Ambiguous

- model name: `Gemma-4-E2B-it`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +2.82, CI [+1.53, +4.12], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_gemma4_e2b_it_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_gemma4_e2b_it_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 68. LocalNewsQA appendix model-gain table | Gemma-4-E2B-it | LocalNewsQA Explicit

- model name: `Gemma-4-E2B-it`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta +0.8471, CI [+0.5000, +1.19], n=17000
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_gemma4_e2b_it_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_gemma4_e2b_it_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 69. LocalNewsQA appendix model-gain table | Gemma-4-E2B-it | LocalNewsQA Exact pair

- model name: `Gemma-4-E2B-it`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +3.47, CI [+2.65, +4.35], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_gemma4_e2b_it_with_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_gemma4_e2b_it_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_gemma4_e2b_it_without_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_gemma4_e2b_it_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 70. LocalNewsQA appendix model-gain table | Gemma-4-E2B-it | LocalNewsQA Margin switch

- model name: `Gemma-4-E2B-it`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +9.82, CI [+8.41, +11.18], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_gemma4_e2b_it_with_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_gemma4_e2b_it_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_gemma4_e2b_it_without_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_gemma4_e2b_it_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 71. LocalNewsQA appendix model-gain table | Gemma-4-E4B-it | LocalNewsQA Overall

- model name: `Gemma-4-E4B-it`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta +2.67, CI [+2.33, +3.01], n=18700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_gemma4_e4b_it_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_gemma4_e4b_it_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 72. LocalNewsQA appendix model-gain table | Gemma-4-E4B-it | LocalNewsQA Ambiguous

- model name: `Gemma-4-E4B-it`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +7.12, CI [+5.53, +8.65], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_gemma4_e4b_it_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_gemma4_e4b_it_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 73. LocalNewsQA appendix model-gain table | Gemma-4-E4B-it | LocalNewsQA Explicit

- model name: `Gemma-4-E4B-it`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta +2.23, CI [+1.88, +2.59], n=17000
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_gemma4_e4b_it_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_gemma4_e4b_it_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 74. LocalNewsQA appendix model-gain table | Gemma-4-E4B-it | LocalNewsQA Exact pair

- model name: `Gemma-4-E4B-it`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +8.76, CI [+7.47, +10.12], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_gemma4_e4b_it_with_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_gemma4_e4b_it_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_gemma4_e4b_it_without_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_gemma4_e4b_it_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 75. LocalNewsQA appendix model-gain table | Gemma-4-E4B-it | LocalNewsQA Margin switch

- model name: `Gemma-4-E4B-it`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +23.18, CI [+21.18, +25.24], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_gemma4_e4b_it_with_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_gemma4_e4b_it_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_gemma4_e4b_it_without_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_gemma4_e4b_it_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 76. LocalNewsQA appendix model-gain table | Mistral-7B-Instruct-v0.2 | LocalNewsQA Overall

- model name: `Mistral-7B-Instruct-v0.2`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta +7.53, CI [+7.05, +8.01], n=18700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_mistral7b_v02_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_mistral7b_v02_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 77. LocalNewsQA appendix model-gain table | Mistral-7B-Instruct-v0.2 | LocalNewsQA Ambiguous

- model name: `Mistral-7B-Instruct-v0.2`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +15.24, CI [+13.24, +17.18], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_mistral7b_v02_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_mistral7b_v02_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 78. LocalNewsQA appendix model-gain table | Mistral-7B-Instruct-v0.2 | LocalNewsQA Explicit

- model name: `Mistral-7B-Instruct-v0.2`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta +6.76, CI [+6.25, +7.24], n=17000
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_mistral7b_v02_with_metadata_target.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_mistral7b_v02_without_metadata_target.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 79. LocalNewsQA appendix model-gain table | Mistral-7B-Instruct-v0.2 | LocalNewsQA Exact pair

- model name: `Mistral-7B-Instruct-v0.2`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +15.41, CI [+13.71, +17.24], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_mistral7b_v02_with_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_mistral7b_v02_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_mistral7b_v02_without_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_mistral7b_v02_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 80. LocalNewsQA appendix model-gain table | Mistral-7B-Instruct-v0.2 | LocalNewsQA Margin switch

- model name: `Mistral-7B-Instruct-v0.2`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +32.29, CI [+30.06, +34.41], n=1700
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_mistral7b_v02_with_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_mistral7b_v02_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/localnewsqa_gold_20260516/external_target/seed_41/localnewsqa_mistral7b_v02_without_metadata_target.jsonl`
  - `results/localnewsqa_gold_20260516/external_contrast/seed_41/localnewsqa_mistral7b_v02_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: ca
CONTINENT: America

TITLE: Facts about the country ca

CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- positive scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with the correct option.
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`
