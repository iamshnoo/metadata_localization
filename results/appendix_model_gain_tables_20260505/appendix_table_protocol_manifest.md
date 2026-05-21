# Appendix Table Protocol Manifest

This file documents every model/dataset cell in the two large appendix model-gain tables.
It is generated from the table CSV artifacts and the source JSONLs used by each cell.

## Reproduction Commands

1. `python src/66_appendix_model_gain_tables.py --seed 41 --bootstrap 2000 --bootstrap-seed 20260505`
2. `python src/67_appendix_table_protocol_manifest.py`

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
- result in table: delta +2.29, CI [+1.87, +2.69], n=35874
- positive source protocol key: `figure9_1b`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=code_grounded; qa_prompt_style=question; answer_cue_style=final_answer_colon; omit_option_labels=False; exact_option_text_instruction=True; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=1.25; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=final_answer_colon; omit_option_labels=False; exact_option_text_instruction=True; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=1.25; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_pretrained_figure9_frozen_full_multiseed/1b_codeg_labels_question_final/seed_41/localnewsqa_eval_target_with_metadata_custom_frozenfig9_1b_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_pretrained_figure9_frozen_full_multiseed/1b_codeg_labels_question_final/seed_41/localnewsqa_eval_target_without_metadata_custom_frozenfig9_1b_tminus_eminus_seed41_c0.jsonl`
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
- result in table: delta +4.37, CI [+3.75, +4.95], n=16409
- positive source protocol key: `figure9_1b`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=code_grounded; qa_prompt_style=question; answer_cue_style=final_answer_colon; omit_option_labels=False; exact_option_text_instruction=True; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=1.25; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=final_answer_colon; omit_option_labels=False; exact_option_text_instruction=True; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=1.25; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_pretrained_figure9_frozen_full_multiseed/1b_codeg_labels_question_final/seed_41/localnewsqa_eval_target_with_metadata_custom_frozenfig9_1b_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_pretrained_figure9_frozen_full_multiseed/1b_codeg_labels_question_final/seed_41/localnewsqa_eval_target_without_metadata_custom_frozenfig9_1b_tminus_eminus_seed41_c0.jsonl`
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
- result in table: delta +0.5394, CI [-0.0103, +1.08], n=19465
- positive source protocol key: `figure9_1b`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=code_grounded; qa_prompt_style=question; answer_cue_style=final_answer_colon; omit_option_labels=False; exact_option_text_instruction=True; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=1.25; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=final_answer_colon; omit_option_labels=False; exact_option_text_instruction=True; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=1.25; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_pretrained_figure9_frozen_full_multiseed/1b_codeg_labels_question_final/seed_41/localnewsqa_eval_target_with_metadata_custom_frozenfig9_1b_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_pretrained_figure9_frozen_full_multiseed/1b_codeg_labels_question_final/seed_41/localnewsqa_eval_target_without_metadata_custom_frozenfig9_1b_tminus_eminus_seed41_c0.jsonl`
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
- result in table: delta +2.64, CI [+2.40, +2.89], n=16409
- positive source protocol key: `figure9_1b`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=code_grounded; qa_prompt_style=question; answer_cue_style=final_answer_colon; omit_option_labels=False; exact_option_text_instruction=True; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=1.25; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=final_answer_colon; omit_option_labels=False; exact_option_text_instruction=True; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=1.25; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_pretrained_figure9_frozen_full_multiseed/1b_codeg_labels_question_final/seed_41/localnewsqa_eval_target_with_metadata_custom_frozenfig9_1b_tplus_eplus_seed41_c0.jsonl`
  - `results/downstream_localnewsqa_pretrained_figure9_contrast_full_multiseed/1b_codeg_labels_question_final/seed_41/localnewsqa_eval_contrast_with_metadata_custom_frozenfig9contrast_1b_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_pretrained_figure9_frozen_full_multiseed/1b_codeg_labels_question_final/seed_41/localnewsqa_eval_target_without_metadata_custom_frozenfig9_1b_tminus_eminus_seed41_c0.jsonl`
  - `results/downstream_localnewsqa_pretrained_figure9_contrast_full_multiseed/1b_codeg_labels_question_final/seed_41/localnewsqa_eval_contrast_without_metadata_custom_frozenfig9contrast_1b_tminus_eminus_seed41_c0.jsonl`
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
- result in table: delta +6.12, CI [+5.75, +6.49], n=16409
- positive source protocol key: `figure9_1b`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=code_grounded; qa_prompt_style=question; answer_cue_style=final_answer_colon; omit_option_labels=False; exact_option_text_instruction=True; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=1.25; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=final_answer_colon; omit_option_labels=False; exact_option_text_instruction=True; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=1.25; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_pretrained_figure9_frozen_full_multiseed/1b_codeg_labels_question_final/seed_41/localnewsqa_eval_target_with_metadata_custom_frozenfig9_1b_tplus_eplus_seed41_c0.jsonl`
  - `results/downstream_localnewsqa_pretrained_figure9_contrast_full_multiseed/1b_codeg_labels_question_final/seed_41/localnewsqa_eval_contrast_with_metadata_custom_frozenfig9contrast_1b_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_pretrained_figure9_frozen_full_multiseed/1b_codeg_labels_question_final/seed_41/localnewsqa_eval_target_without_metadata_custom_frozenfig9_1b_tminus_eminus_seed41_c0.jsonl`
  - `results/downstream_localnewsqa_pretrained_figure9_contrast_full_multiseed/1b_codeg_labels_question_final/seed_41/localnewsqa_eval_contrast_without_metadata_custom_frozenfig9contrast_1b_tminus_eminus_seed41_c0.jsonl`
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
- result in table: delta +2.30, CI [+1.76, +2.86], n=35874
- positive source protocol key: `sft_1b_lnqa`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_plain; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.25; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.25; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_sft_figure9_1b_checkpoint_nameplain_countrycue_beta025_multiseed_contrib2/ckpt_7764/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_sft_figure9_1b_checkpoint_nameplain_countrycue_beta025_multiseed_contrib2/ckpt_7764/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tminus_eminus_seed41_c0.jsonl`
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
- result in table: delta +4.39, CI [+3.58, +5.21], n=16409
- positive source protocol key: `sft_1b_lnqa`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_plain; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.25; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.25; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_sft_figure9_1b_checkpoint_nameplain_countrycue_beta025_multiseed_contrib2/ckpt_7764/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_sft_figure9_1b_checkpoint_nameplain_countrycue_beta025_multiseed_contrib2/ckpt_7764/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tminus_eminus_seed41_c0.jsonl`
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
- result in table: delta +0.5343, CI [-0.2313, +1.27], n=19465
- positive source protocol key: `sft_1b_lnqa`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_plain; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.25; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.25; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_sft_figure9_1b_checkpoint_nameplain_countrycue_beta025_multiseed_contrib2/ckpt_7764/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_sft_figure9_1b_checkpoint_nameplain_countrycue_beta025_multiseed_contrib2/ckpt_7764/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tminus_eminus_seed41_c0.jsonl`
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
- result in table: delta +3.80, CI [+3.52, +4.13], n=16409
- positive source protocol key: `sft_1b_lnqa`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_plain; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.25; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.25; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_sft_figure9_1b_checkpoint_nameplain_countrycue_beta025_multiseed_contrib2/ckpt_7764/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tplus_eplus_seed41_c0.jsonl`
  - `results/downstream_localnewsqa_sft_figure9_1b_checkpoint_nameplain_countrycue_beta025_contrast_multiseed/ckpt_7764/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41/localnewsqa_eval_contrast_with_metadata_custom_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_sft_figure9_1b_checkpoint_nameplain_countrycue_beta025_multiseed_contrib2/ckpt_7764/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tminus_eminus_seed41_c0.jsonl`
  - `results/downstream_localnewsqa_sft_figure9_1b_checkpoint_nameplain_countrycue_beta025_contrast_multiseed/ckpt_7764/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41/localnewsqa_eval_contrast_without_metadata_custom_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tminus_eminus_seed41_c0.jsonl`
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
- result in table: delta +9.32, CI [+8.87, +9.80], n=16409
- positive source protocol key: `sft_1b_lnqa`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_plain; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.25; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.25; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_sft_figure9_1b_checkpoint_nameplain_countrycue_beta025_multiseed_contrib2/ckpt_7764/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tplus_eplus_seed41_c0.jsonl`
  - `results/downstream_localnewsqa_sft_figure9_1b_checkpoint_nameplain_countrycue_beta025_contrast_multiseed/ckpt_7764/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41/localnewsqa_eval_contrast_with_metadata_custom_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_sft_figure9_1b_checkpoint_nameplain_countrycue_beta025_multiseed_contrib2/ckpt_7764/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tminus_eminus_seed41_c0.jsonl`
  - `results/downstream_localnewsqa_sft_figure9_1b_checkpoint_nameplain_countrycue_beta025_contrast_multiseed/ckpt_7764/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41/localnewsqa_eval_contrast_without_metadata_custom_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tminus_eminus_seed41_c0.jsonl`
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
- result in table: delta +4.98, CI [+4.36, +5.62], n=35874
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_grounded; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_pretrained_figure9_frozen_full_multiseed/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_frozenfig9_3b_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_pretrained_figure9_frozen_full_multiseed/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_frozenfig9_3b_tminus_eminus_seed41_c0.jsonl`
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
- result in table: delta +7.91, CI [+6.95, +8.86], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_grounded; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_pretrained_figure9_frozen_full_multiseed/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_frozenfig9_3b_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_pretrained_figure9_frozen_full_multiseed/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_frozenfig9_3b_tminus_eminus_seed41_c0.jsonl`
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
- result in table: delta +2.51, CI [+1.71, +3.35], n=19465
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_grounded; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_pretrained_figure9_frozen_full_multiseed/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_frozenfig9_3b_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_pretrained_figure9_frozen_full_multiseed/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_frozenfig9_3b_tminus_eminus_seed41_c0.jsonl`
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
- result in table: delta +6.36, CI [+5.98, +6.74], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_grounded; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_pretrained_figure9_frozen_full_multiseed/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_frozenfig9_3b_tplus_eplus_seed41_c0.jsonl`
  - `results/downstream_localnewsqa_pretrained_figure9_contrast_full_multiseed/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_contrast_with_metadata_custom_frozenfig9contrast_3b_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_pretrained_figure9_frozen_full_multiseed/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_frozenfig9_3b_tminus_eminus_seed41_c0.jsonl`
  - `results/downstream_localnewsqa_pretrained_figure9_contrast_full_multiseed/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_contrast_without_metadata_custom_frozenfig9contrast_3b_tminus_eminus_seed41_c0.jsonl`
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
- result in table: delta +15.39, CI [+14.85, +15.91], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_grounded; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_pretrained_figure9_frozen_full_multiseed/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_frozenfig9_3b_tplus_eplus_seed41_c0.jsonl`
  - `results/downstream_localnewsqa_pretrained_figure9_contrast_full_multiseed/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_contrast_with_metadata_custom_frozenfig9contrast_3b_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_pretrained_figure9_frozen_full_multiseed/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_frozenfig9_3b_tminus_eminus_seed41_c0.jsonl`
  - `results/downstream_localnewsqa_pretrained_figure9_contrast_full_multiseed/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_contrast_without_metadata_custom_frozenfig9contrast_3b_tminus_eminus_seed41_c0.jsonl`
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
- result in table: delta +4.92, CI [+4.32, +5.52], n=35874
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_grounded; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_sft_figure9_full_multiseed/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_sftfig9_3b_best3b_chat_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_sft_figure9_full_multiseed/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_sftfig9_3b_best3b_chat_tminus_eminus_seed41_c0.jsonl`
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
- result in table: delta +8.84, CI [+7.95, +9.78], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_grounded; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_sft_figure9_full_multiseed/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_sftfig9_3b_best3b_chat_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_sft_figure9_full_multiseed/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_sftfig9_3b_best3b_chat_tminus_eminus_seed41_c0.jsonl`
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
- result in table: delta +1.61, CI [+0.8373, +2.41], n=19465
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_grounded; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_sft_figure9_full_multiseed/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_sftfig9_3b_best3b_chat_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_sft_figure9_full_multiseed/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_sftfig9_3b_best3b_chat_tminus_eminus_seed41_c0.jsonl`
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
- result in table: delta +4.58, CI [+4.17, +4.98], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_grounded; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_sft_figure9_full_multiseed/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_sftfig9_3b_best3b_chat_tplus_eplus_seed41_c0.jsonl`
  - `results/downstream_localnewsqa_sft_figure9_contrast_full_multiseed/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_contrast_with_metadata_custom_sftfig9_3b_best3b_chat_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_sft_figure9_full_multiseed/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_sftfig9_3b_best3b_chat_tminus_eminus_seed41_c0.jsonl`
  - `results/downstream_localnewsqa_sft_figure9_contrast_full_multiseed/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_contrast_without_metadata_custom_sftfig9_3b_best3b_chat_tminus_eminus_seed41_c0.jsonl`
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
- result in table: delta +9.21, CI [+8.63, +9.79], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_grounded; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_sft_figure9_full_multiseed/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_sftfig9_3b_best3b_chat_tplus_eplus_seed41_c0.jsonl`
  - `results/downstream_localnewsqa_sft_figure9_contrast_full_multiseed/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_contrast_with_metadata_custom_sftfig9_3b_best3b_chat_tplus_eplus_seed41_c0.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_sft_figure9_full_multiseed/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_sftfig9_3b_best3b_chat_tminus_eminus_seed41_c0.jsonl`
  - `results/downstream_localnewsqa_sft_figure9_contrast_full_multiseed/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_contrast_without_metadata_custom_sftfig9_3b_best3b_chat_tminus_eminus_seed41_c0.jsonl`
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

### 21. LocalNewsQA appendix model-gain table | LLaMA-3.2-1B Base | LocalNewsQA Overall

- model name: `LLaMA-3.2-1B Base`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta +0.1004, CI [-0.2203, +0.4209], n=35874
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/llama32_1b/base/seed_41/target/localnewsqa_llama32_1b_base_with_metadata_target_matrix_llama32_1b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/llama32_1b/base/seed_41/target/localnewsqa_llama32_1b_base_without_metadata_target_matrix_llama32_1b_base_seed41.jsonl`
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

### 22. LocalNewsQA appendix model-gain table | LLaMA-3.2-1B Base | LocalNewsQA Ambiguous

- model name: `LLaMA-3.2-1B Base`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +2.08, CI [+1.54, +2.61], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/llama32_1b/base/seed_41/target/localnewsqa_llama32_1b_base_with_metadata_target_matrix_llama32_1b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/llama32_1b/base/seed_41/target/localnewsqa_llama32_1b_base_without_metadata_target_matrix_llama32_1b_base_seed41.jsonl`
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

### 23. LocalNewsQA appendix model-gain table | LLaMA-3.2-1B Base | LocalNewsQA Explicit

- model name: `LLaMA-3.2-1B Base`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta -1.57, CI [-1.98, -1.16], n=19465
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/llama32_1b/base/seed_41/target/localnewsqa_llama32_1b_base_with_metadata_target_matrix_llama32_1b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/llama32_1b/base/seed_41/target/localnewsqa_llama32_1b_base_without_metadata_target_matrix_llama32_1b_base_seed41.jsonl`
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

### 24. LocalNewsQA appendix model-gain table | LLaMA-3.2-1B Base | LocalNewsQA Exact pair

- model name: `LLaMA-3.2-1B Base`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +2.75, CI [+2.51, +3.00], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/llama32_1b/base/seed_41/target/localnewsqa_llama32_1b_base_with_metadata_target_matrix_llama32_1b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/llama32_1b/base/seed_41/contrast/localnewsqa_llama32_1b_base_with_metadata_contrast_matrix_llama32_1b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/llama32_1b/base/seed_41/target/localnewsqa_llama32_1b_base_without_metadata_target_matrix_llama32_1b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/llama32_1b/base/seed_41/contrast/localnewsqa_llama32_1b_base_without_metadata_contrast_matrix_llama32_1b_base_seed41.jsonl`
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

### 25. LocalNewsQA appendix model-gain table | LLaMA-3.2-1B Base | LocalNewsQA Margin switch

- model name: `LLaMA-3.2-1B Base`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +6.52, CI [+6.16, +6.92], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/llama32_1b/base/seed_41/target/localnewsqa_llama32_1b_base_with_metadata_target_matrix_llama32_1b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/llama32_1b/base/seed_41/contrast/localnewsqa_llama32_1b_base_with_metadata_contrast_matrix_llama32_1b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/llama32_1b/base/seed_41/target/localnewsqa_llama32_1b_base_without_metadata_target_matrix_llama32_1b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/llama32_1b/base/seed_41/contrast/localnewsqa_llama32_1b_base_without_metadata_contrast_matrix_llama32_1b_base_seed41.jsonl`
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

### 26. LocalNewsQA appendix model-gain table | LLaMA-3.2-1B Inst. | LocalNewsQA Overall

- model name: `LLaMA-3.2-1B Inst.`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta -0.7359, CI [-1.05, -0.4264], n=35874
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_llama32_1b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_llama32_1b_without_metadata_target.jsonl`
- fixed-example positive prompt:
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

### 27. LocalNewsQA appendix model-gain table | LLaMA-3.2-1B Inst. | LocalNewsQA Ambiguous

- model name: `LLaMA-3.2-1B Inst.`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +1.68, CI [+1.22, +2.18], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_llama32_1b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_llama32_1b_without_metadata_target.jsonl`
- fixed-example positive prompt:
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

### 28. LocalNewsQA appendix model-gain table | LLaMA-3.2-1B Inst. | LocalNewsQA Explicit

- model name: `LLaMA-3.2-1B Inst.`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta -2.77, CI [-3.16, -2.37], n=19465
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_llama32_1b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_llama32_1b_without_metadata_target.jsonl`
- fixed-example positive prompt:
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

### 29. LocalNewsQA appendix model-gain table | LLaMA-3.2-1B Inst. | LocalNewsQA Exact pair

- model name: `LLaMA-3.2-1B Inst.`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +2.46, CI [+2.19, +2.75], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_llama32_1b_with_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_llama32_1b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_llama32_1b_without_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_llama32_1b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
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

### 30. LocalNewsQA appendix model-gain table | LLaMA-3.2-1B Inst. | LocalNewsQA Margin switch

- model name: `LLaMA-3.2-1B Inst.`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +5.57, CI [+5.16, +5.97], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_llama32_1b_with_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_llama32_1b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_llama32_1b_without_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_llama32_1b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
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

### 31. LocalNewsQA appendix model-gain table | LLaMA-3.2-3B Base | LocalNewsQA Overall

- model name: `LLaMA-3.2-3B Base`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta +1.11, CI [+0.7610, +1.44], n=35874
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/llama32_3b/base/seed_41/target/localnewsqa_llama32_3b_base_with_metadata_target_matrix_llama32_3b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/llama32_3b/base/seed_41/target/localnewsqa_llama32_3b_base_without_metadata_target_matrix_llama32_3b_base_seed41.jsonl`
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

### 32. LocalNewsQA appendix model-gain table | LLaMA-3.2-3B Base | LocalNewsQA Ambiguous

- model name: `LLaMA-3.2-3B Base`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +5.00, CI [+4.47, +5.53], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/llama32_3b/base/seed_41/target/localnewsqa_llama32_3b_base_with_metadata_target_matrix_llama32_3b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/llama32_3b/base/seed_41/target/localnewsqa_llama32_3b_base_without_metadata_target_matrix_llama32_3b_base_seed41.jsonl`
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

### 33. LocalNewsQA appendix model-gain table | LLaMA-3.2-3B Base | LocalNewsQA Explicit

- model name: `LLaMA-3.2-3B Base`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta -2.18, CI [-2.62, -1.75], n=19465
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/llama32_3b/base/seed_41/target/localnewsqa_llama32_3b_base_with_metadata_target_matrix_llama32_3b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/llama32_3b/base/seed_41/target/localnewsqa_llama32_3b_base_without_metadata_target_matrix_llama32_3b_base_seed41.jsonl`
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

### 34. LocalNewsQA appendix model-gain table | LLaMA-3.2-3B Base | LocalNewsQA Exact pair

- model name: `LLaMA-3.2-3B Base`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +5.86, CI [+5.49, +6.22], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/llama32_3b/base/seed_41/target/localnewsqa_llama32_3b_base_with_metadata_target_matrix_llama32_3b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/llama32_3b/base/seed_41/contrast/localnewsqa_llama32_3b_base_with_metadata_contrast_matrix_llama32_3b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/llama32_3b/base/seed_41/target/localnewsqa_llama32_3b_base_without_metadata_target_matrix_llama32_3b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/llama32_3b/base/seed_41/contrast/localnewsqa_llama32_3b_base_without_metadata_contrast_matrix_llama32_3b_base_seed41.jsonl`
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

### 35. LocalNewsQA appendix model-gain table | LLaMA-3.2-3B Base | LocalNewsQA Margin switch

- model name: `LLaMA-3.2-3B Base`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +12.08, CI [+11.56, +12.60], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/llama32_3b/base/seed_41/target/localnewsqa_llama32_3b_base_with_metadata_target_matrix_llama32_3b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/llama32_3b/base/seed_41/contrast/localnewsqa_llama32_3b_base_with_metadata_contrast_matrix_llama32_3b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/llama32_3b/base/seed_41/target/localnewsqa_llama32_3b_base_without_metadata_target_matrix_llama32_3b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/llama32_3b/base/seed_41/contrast/localnewsqa_llama32_3b_base_without_metadata_contrast_matrix_llama32_3b_base_seed41.jsonl`
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

### 36. LocalNewsQA appendix model-gain table | LLaMA-3.2-3B Inst. | LocalNewsQA Overall

- model name: `LLaMA-3.2-3B Inst.`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta -4.64, CI [-5.01, -4.25], n=35874
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_llama32_3b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_llama32_3b_without_metadata_target.jsonl`
- fixed-example positive prompt:
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

### 37. LocalNewsQA appendix model-gain table | LLaMA-3.2-3B Inst. | LocalNewsQA Ambiguous

- model name: `LLaMA-3.2-3B Inst.`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta -0.0183, CI [-0.5972, +0.5485], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_llama32_3b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_llama32_3b_without_metadata_target.jsonl`
- fixed-example positive prompt:
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

### 38. LocalNewsQA appendix model-gain table | LLaMA-3.2-3B Inst. | LocalNewsQA Explicit

- model name: `LLaMA-3.2-3B Inst.`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta -8.53, CI [-9.03, -8.05], n=19465
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_llama32_3b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_llama32_3b_without_metadata_target.jsonl`
- fixed-example positive prompt:
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

### 39. LocalNewsQA appendix model-gain table | LLaMA-3.2-3B Inst. | LocalNewsQA Exact pair

- model name: `LLaMA-3.2-3B Inst.`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +3.95, CI [+3.64, +4.25], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_llama32_3b_with_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_llama32_3b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_llama32_3b_without_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_llama32_3b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
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

### 40. LocalNewsQA appendix model-gain table | LLaMA-3.2-3B Inst. | LocalNewsQA Margin switch

- model name: `LLaMA-3.2-3B Inst.`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +8.65, CI [+8.22, +9.10], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_llama32_3b_with_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_llama32_3b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_llama32_3b_without_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_llama32_3b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
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

### 41. LocalNewsQA appendix model-gain table | Qwen2.5-1.5B Base | LocalNewsQA Overall

- model name: `Qwen2.5-1.5B Base`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta +0.9673, CI [+0.6494, +1.27], n=35874
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen25_1p5b/base/seed_41/target/localnewsqa_qwen25_1p5b_base_with_metadata_target_matrix_qwen25_1p5b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen25_1p5b/base/seed_41/target/localnewsqa_qwen25_1p5b_base_without_metadata_target_matrix_qwen25_1p5b_base_seed41.jsonl`
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

### 42. LocalNewsQA appendix model-gain table | Qwen2.5-1.5B Base | LocalNewsQA Ambiguous

- model name: `Qwen2.5-1.5B Base`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +2.69, CI [+2.22, +3.15], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen25_1p5b/base/seed_41/target/localnewsqa_qwen25_1p5b_base_with_metadata_target_matrix_qwen25_1p5b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen25_1p5b/base/seed_41/target/localnewsqa_qwen25_1p5b_base_without_metadata_target_matrix_qwen25_1p5b_base_seed41.jsonl`
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

### 43. LocalNewsQA appendix model-gain table | Qwen2.5-1.5B Base | LocalNewsQA Explicit

- model name: `Qwen2.5-1.5B Base`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta -0.4881, CI [-0.8940, -0.0923], n=19465
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen25_1p5b/base/seed_41/target/localnewsqa_qwen25_1p5b_base_with_metadata_target_matrix_qwen25_1p5b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen25_1p5b/base/seed_41/target/localnewsqa_qwen25_1p5b_base_without_metadata_target_matrix_qwen25_1p5b_base_seed41.jsonl`
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

### 44. LocalNewsQA appendix model-gain table | Qwen2.5-1.5B Base | LocalNewsQA Exact pair

- model name: `Qwen2.5-1.5B Base`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +2.07, CI [+1.85, +2.29], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen25_1p5b/base/seed_41/target/localnewsqa_qwen25_1p5b_base_with_metadata_target_matrix_qwen25_1p5b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/qwen25_1p5b/base/seed_41/contrast/localnewsqa_qwen25_1p5b_base_with_metadata_contrast_matrix_qwen25_1p5b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen25_1p5b/base/seed_41/target/localnewsqa_qwen25_1p5b_base_without_metadata_target_matrix_qwen25_1p5b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/qwen25_1p5b/base/seed_41/contrast/localnewsqa_qwen25_1p5b_base_without_metadata_contrast_matrix_qwen25_1p5b_base_seed41.jsonl`
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

### 45. LocalNewsQA appendix model-gain table | Qwen2.5-1.5B Base | LocalNewsQA Margin switch

- model name: `Qwen2.5-1.5B Base`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +4.09, CI [+3.78, +4.39], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen25_1p5b/base/seed_41/target/localnewsqa_qwen25_1p5b_base_with_metadata_target_matrix_qwen25_1p5b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/qwen25_1p5b/base/seed_41/contrast/localnewsqa_qwen25_1p5b_base_with_metadata_contrast_matrix_qwen25_1p5b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen25_1p5b/base/seed_41/target/localnewsqa_qwen25_1p5b_base_without_metadata_target_matrix_qwen25_1p5b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/qwen25_1p5b/base/seed_41/contrast/localnewsqa_qwen25_1p5b_base_without_metadata_contrast_matrix_qwen25_1p5b_base_seed41.jsonl`
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

### 46. LocalNewsQA appendix model-gain table | Qwen2.5-1.5B Inst. | LocalNewsQA Overall

- model name: `Qwen2.5-1.5B Inst.`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta +2.16, CI [+1.79, +2.54], n=35874
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen25_1p5b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen25_1p5b_without_metadata_target.jsonl`
- fixed-example positive prompt:
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

### 47. LocalNewsQA appendix model-gain table | Qwen2.5-1.5B Inst. | LocalNewsQA Ambiguous

- model name: `Qwen2.5-1.5B Inst.`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +8.47, CI [+7.86, +9.04], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen25_1p5b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen25_1p5b_without_metadata_target.jsonl`
- fixed-example positive prompt:
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

### 48. LocalNewsQA appendix model-gain table | Qwen2.5-1.5B Inst. | LocalNewsQA Explicit

- model name: `Qwen2.5-1.5B Inst.`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta -3.16, CI [-3.63, -2.74], n=19465
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen25_1p5b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen25_1p5b_without_metadata_target.jsonl`
- fixed-example positive prompt:
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

### 49. LocalNewsQA appendix model-gain table | Qwen2.5-1.5B Inst. | LocalNewsQA Exact pair

- model name: `Qwen2.5-1.5B Inst.`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +9.54, CI [+9.10, +9.98], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen25_1p5b_with_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_qwen25_1p5b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen25_1p5b_without_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_qwen25_1p5b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
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

### 50. LocalNewsQA appendix model-gain table | Qwen2.5-1.5B Inst. | LocalNewsQA Margin switch

- model name: `Qwen2.5-1.5B Inst.`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +17.44, CI [+16.87, +18.02], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen25_1p5b_with_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_qwen25_1p5b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen25_1p5b_without_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_qwen25_1p5b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
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

### 51. LocalNewsQA appendix model-gain table | Qwen2.5-3B Base | LocalNewsQA Overall

- model name: `Qwen2.5-3B Base`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta +1.48, CI [+1.11, +1.85], n=35874
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen25_3b/base/seed_41/target/localnewsqa_qwen25_3b_base_with_metadata_target_matrix_qwen25_3b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen25_3b/base/seed_41/target/localnewsqa_qwen25_3b_base_without_metadata_target_matrix_qwen25_3b_base_seed41.jsonl`
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

### 52. LocalNewsQA appendix model-gain table | Qwen2.5-3B Base | LocalNewsQA Ambiguous

- model name: `Qwen2.5-3B Base`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +8.06, CI [+7.43, +8.63], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen25_3b/base/seed_41/target/localnewsqa_qwen25_3b_base_with_metadata_target_matrix_qwen25_3b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen25_3b/base/seed_41/target/localnewsqa_qwen25_3b_base_without_metadata_target_matrix_qwen25_3b_base_seed41.jsonl`
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

### 53. LocalNewsQA appendix model-gain table | Qwen2.5-3B Base | LocalNewsQA Explicit

- model name: `Qwen2.5-3B Base`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta -4.06, CI [-4.48, -3.62], n=19465
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen25_3b/base/seed_41/target/localnewsqa_qwen25_3b_base_with_metadata_target_matrix_qwen25_3b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen25_3b/base/seed_41/target/localnewsqa_qwen25_3b_base_without_metadata_target_matrix_qwen25_3b_base_seed41.jsonl`
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

### 54. LocalNewsQA appendix model-gain table | Qwen2.5-3B Base | LocalNewsQA Exact pair

- model name: `Qwen2.5-3B Base`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +11.09, CI [+10.60, +11.59], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen25_3b/base/seed_41/target/localnewsqa_qwen25_3b_base_with_metadata_target_matrix_qwen25_3b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/qwen25_3b/base/seed_41/contrast/localnewsqa_qwen25_3b_base_with_metadata_contrast_matrix_qwen25_3b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen25_3b/base/seed_41/target/localnewsqa_qwen25_3b_base_without_metadata_target_matrix_qwen25_3b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/qwen25_3b/base/seed_41/contrast/localnewsqa_qwen25_3b_base_without_metadata_contrast_matrix_qwen25_3b_base_seed41.jsonl`
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

### 55. LocalNewsQA appendix model-gain table | Qwen2.5-3B Base | LocalNewsQA Margin switch

- model name: `Qwen2.5-3B Base`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +19.73, CI [+19.14, +20.35], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen25_3b/base/seed_41/target/localnewsqa_qwen25_3b_base_with_metadata_target_matrix_qwen25_3b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/qwen25_3b/base/seed_41/contrast/localnewsqa_qwen25_3b_base_with_metadata_contrast_matrix_qwen25_3b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen25_3b/base/seed_41/target/localnewsqa_qwen25_3b_base_without_metadata_target_matrix_qwen25_3b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/qwen25_3b/base/seed_41/contrast/localnewsqa_qwen25_3b_base_without_metadata_contrast_matrix_qwen25_3b_base_seed41.jsonl`
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

### 56. LocalNewsQA appendix model-gain table | Qwen2.5-3B Inst. | LocalNewsQA Overall

- model name: `Qwen2.5-3B Inst.`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta +3.41, CI [+3.09, +3.74], n=35874
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen25_3b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen25_3b_without_metadata_target.jsonl`
- fixed-example positive prompt:
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

### 57. LocalNewsQA appendix model-gain table | Qwen2.5-3B Inst. | LocalNewsQA Ambiguous

- model name: `Qwen2.5-3B Inst.`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +8.41, CI [+7.87, +9.00], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen25_3b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen25_3b_without_metadata_target.jsonl`
- fixed-example positive prompt:
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

### 58. LocalNewsQA appendix model-gain table | Qwen2.5-3B Inst. | LocalNewsQA Explicit

- model name: `Qwen2.5-3B Inst.`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta -0.7963, CI [-1.15, -0.4264], n=19465
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen25_3b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen25_3b_without_metadata_target.jsonl`
- fixed-example positive prompt:
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

### 59. LocalNewsQA appendix model-gain table | Qwen2.5-3B Inst. | LocalNewsQA Exact pair

- model name: `Qwen2.5-3B Inst.`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +9.84, CI [+9.37, +10.32], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen25_3b_with_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_qwen25_3b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen25_3b_without_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_qwen25_3b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
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

### 60. LocalNewsQA appendix model-gain table | Qwen2.5-3B Inst. | LocalNewsQA Margin switch

- model name: `Qwen2.5-3B Inst.`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +19.33, CI [+18.73, +19.92], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen25_3b_with_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_qwen25_3b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen25_3b_without_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_qwen25_3b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
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

### 61. LocalNewsQA appendix model-gain table | Qwen3.5-0.8B Base | LocalNewsQA Overall

- model name: `Qwen3.5-0.8B Base`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta +0.9645, CI [+0.6829, +1.24], n=35874
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_0p8b/base/seed_41/target/localnewsqa_qwen35_0p8b_base_with_metadata_target_matrix_qwen35_0p8b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_0p8b/base/seed_41/target/localnewsqa_qwen35_0p8b_base_without_metadata_target_matrix_qwen35_0p8b_base_seed41.jsonl`
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

### 62. LocalNewsQA appendix model-gain table | Qwen3.5-0.8B Base | LocalNewsQA Ambiguous

- model name: `Qwen3.5-0.8B Base`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +3.89, CI [+3.44, +4.36], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_0p8b/base/seed_41/target/localnewsqa_qwen35_0p8b_base_with_metadata_target_matrix_qwen35_0p8b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_0p8b/base/seed_41/target/localnewsqa_qwen35_0p8b_base_without_metadata_target_matrix_qwen35_0p8b_base_seed41.jsonl`
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

### 63. LocalNewsQA appendix model-gain table | Qwen3.5-0.8B Base | LocalNewsQA Explicit

- model name: `Qwen3.5-0.8B Base`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta -1.51, CI [-1.86, -1.16], n=19465
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_0p8b/base/seed_41/target/localnewsqa_qwen35_0p8b_base_with_metadata_target_matrix_qwen35_0p8b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_0p8b/base/seed_41/target/localnewsqa_qwen35_0p8b_base_without_metadata_target_matrix_qwen35_0p8b_base_seed41.jsonl`
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

### 64. LocalNewsQA appendix model-gain table | Qwen3.5-0.8B Base | LocalNewsQA Exact pair

- model name: `Qwen3.5-0.8B Base`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +3.83, CI [+3.53, +4.13], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_0p8b/base/seed_41/target/localnewsqa_qwen35_0p8b_base_with_metadata_target_matrix_qwen35_0p8b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/qwen35_0p8b/base/seed_41/contrast/localnewsqa_qwen35_0p8b_base_with_metadata_contrast_matrix_qwen35_0p8b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_0p8b/base/seed_41/target/localnewsqa_qwen35_0p8b_base_without_metadata_target_matrix_qwen35_0p8b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/qwen35_0p8b/base/seed_41/contrast/localnewsqa_qwen35_0p8b_base_without_metadata_contrast_matrix_qwen35_0p8b_base_seed41.jsonl`
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

### 65. LocalNewsQA appendix model-gain table | Qwen3.5-0.8B Base | LocalNewsQA Margin switch

- model name: `Qwen3.5-0.8B Base`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +7.98, CI [+7.58, +8.41], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_0p8b/base/seed_41/target/localnewsqa_qwen35_0p8b_base_with_metadata_target_matrix_qwen35_0p8b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/qwen35_0p8b/base/seed_41/contrast/localnewsqa_qwen35_0p8b_base_with_metadata_contrast_matrix_qwen35_0p8b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_0p8b/base/seed_41/target/localnewsqa_qwen35_0p8b_base_without_metadata_target_matrix_qwen35_0p8b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/qwen35_0p8b/base/seed_41/contrast/localnewsqa_qwen35_0p8b_base_without_metadata_contrast_matrix_qwen35_0p8b_base_seed41.jsonl`
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

### 66. LocalNewsQA appendix model-gain table | Qwen3.5-0.8B Chat | LocalNewsQA Overall

- model name: `Qwen3.5-0.8B Chat`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta +0.5742, CI [+0.2815, +0.8474], n=35874
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_0p8b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_0p8b_without_metadata_target.jsonl`
- fixed-example positive prompt:
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

### 67. LocalNewsQA appendix model-gain table | Qwen3.5-0.8B Chat | LocalNewsQA Ambiguous

- model name: `Qwen3.5-0.8B Chat`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +3.50, CI [+3.07, +3.98], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_0p8b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_0p8b_without_metadata_target.jsonl`
- fixed-example positive prompt:
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

### 68. LocalNewsQA appendix model-gain table | Qwen3.5-0.8B Chat | LocalNewsQA Explicit

- model name: `Qwen3.5-0.8B Chat`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta -1.90, CI [-2.25, -1.56], n=19465
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_0p8b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_0p8b_without_metadata_target.jsonl`
- fixed-example positive prompt:
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

### 69. LocalNewsQA appendix model-gain table | Qwen3.5-0.8B Chat | LocalNewsQA Exact pair

- model name: `Qwen3.5-0.8B Chat`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +3.32, CI [+3.05, +3.59], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_0p8b_with_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_qwen35_0p8b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_0p8b_without_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_qwen35_0p8b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
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

### 70. LocalNewsQA appendix model-gain table | Qwen3.5-0.8B Chat | LocalNewsQA Margin switch

- model name: `Qwen3.5-0.8B Chat`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +7.32, CI [+6.92, +7.71], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_0p8b_with_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_qwen35_0p8b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_0p8b_without_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_qwen35_0p8b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
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

### 71. LocalNewsQA appendix model-gain table | Qwen3.5-2B Base | LocalNewsQA Overall

- model name: `Qwen3.5-2B Base`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta +4.11, CI [+3.83, +4.41], n=35874
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_2b/base/seed_41/target/localnewsqa_qwen35_2b_base_with_metadata_target_matrix_qwen35_2b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_2b/base/seed_41/target/localnewsqa_qwen35_2b_base_without_metadata_target_matrix_qwen35_2b_base_seed41.jsonl`
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

### 72. LocalNewsQA appendix model-gain table | Qwen3.5-2B Base | LocalNewsQA Ambiguous

- model name: `Qwen3.5-2B Base`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +9.05, CI [+8.52, +9.57], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_2b/base/seed_41/target/localnewsqa_qwen35_2b_base_with_metadata_target_matrix_qwen35_2b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_2b/base/seed_41/target/localnewsqa_qwen35_2b_base_without_metadata_target_matrix_qwen35_2b_base_seed41.jsonl`
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

### 73. LocalNewsQA appendix model-gain table | Qwen3.5-2B Base | LocalNewsQA Explicit

- model name: `Qwen3.5-2B Base`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta -0.0565, CI [-0.3802, +0.2569], n=19465
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_2b/base/seed_41/target/localnewsqa_qwen35_2b_base_with_metadata_target_matrix_qwen35_2b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_2b/base/seed_41/target/localnewsqa_qwen35_2b_base_without_metadata_target_matrix_qwen35_2b_base_seed41.jsonl`
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

### 74. LocalNewsQA appendix model-gain table | Qwen3.5-2B Base | LocalNewsQA Exact pair

- model name: `Qwen3.5-2B Base`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +9.57, CI [+9.13, +10.01], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_2b/base/seed_41/target/localnewsqa_qwen35_2b_base_with_metadata_target_matrix_qwen35_2b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/qwen35_2b/base/seed_41/contrast/localnewsqa_qwen35_2b_base_with_metadata_contrast_matrix_qwen35_2b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_2b/base/seed_41/target/localnewsqa_qwen35_2b_base_without_metadata_target_matrix_qwen35_2b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/qwen35_2b/base/seed_41/contrast/localnewsqa_qwen35_2b_base_without_metadata_contrast_matrix_qwen35_2b_base_seed41.jsonl`
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

### 75. LocalNewsQA appendix model-gain table | Qwen3.5-2B Base | LocalNewsQA Margin switch

- model name: `Qwen3.5-2B Base`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +16.83, CI [+16.24, +17.41], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_2b/base/seed_41/target/localnewsqa_qwen35_2b_base_with_metadata_target_matrix_qwen35_2b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/qwen35_2b/base/seed_41/contrast/localnewsqa_qwen35_2b_base_with_metadata_contrast_matrix_qwen35_2b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_2b/base/seed_41/target/localnewsqa_qwen35_2b_base_without_metadata_target_matrix_qwen35_2b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/qwen35_2b/base/seed_41/contrast/localnewsqa_qwen35_2b_base_without_metadata_contrast_matrix_qwen35_2b_base_seed41.jsonl`
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

### 76. LocalNewsQA appendix model-gain table | Qwen3.5-2B Chat | LocalNewsQA Overall

- model name: `Qwen3.5-2B Chat`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta +4.89, CI [+4.56, +5.23], n=35874
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_2b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_2b_without_metadata_target.jsonl`
- fixed-example positive prompt:
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

### 77. LocalNewsQA appendix model-gain table | Qwen3.5-2B Chat | LocalNewsQA Ambiguous

- model name: `Qwen3.5-2B Chat`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +8.51, CI [+7.95, +9.05], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_2b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_2b_without_metadata_target.jsonl`
- fixed-example positive prompt:
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

### 78. LocalNewsQA appendix model-gain table | Qwen3.5-2B Chat | LocalNewsQA Explicit

- model name: `Qwen3.5-2B Chat`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta +1.84, CI [+1.44, +2.25], n=19465
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_2b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_2b_without_metadata_target.jsonl`
- fixed-example positive prompt:
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

### 79. LocalNewsQA appendix model-gain table | Qwen3.5-2B Chat | LocalNewsQA Exact pair

- model name: `Qwen3.5-2B Chat`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +8.85, CI [+8.42, +9.31], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_2b_with_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_qwen35_2b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_2b_without_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_qwen35_2b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
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

### 80. LocalNewsQA appendix model-gain table | Qwen3.5-2B Chat | LocalNewsQA Margin switch

- model name: `Qwen3.5-2B Chat`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +16.00, CI [+15.45, +16.55], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_2b_with_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_qwen35_2b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_2b_without_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_qwen35_2b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
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

### 81. LocalNewsQA appendix model-gain table | Qwen3.5-4B Base | LocalNewsQA Overall

- model name: `Qwen3.5-4B Base`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta +0.7275, CI [+0.5631, +0.8892], n=35874
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_4b/base/seed_41/target/localnewsqa_qwen35_4b_base_with_metadata_target_matrix_qwen35_4b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_4b/base/seed_41/target/localnewsqa_qwen35_4b_base_without_metadata_target_matrix_qwen35_4b_base_seed41.jsonl`
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

### 82. LocalNewsQA appendix model-gain table | Qwen3.5-4B Base | LocalNewsQA Ambiguous

- model name: `Qwen3.5-4B Base`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +1.56, CI [+1.29, +1.85], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_4b/base/seed_41/target/localnewsqa_qwen35_4b_base_with_metadata_target_matrix_qwen35_4b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_4b/base/seed_41/target/localnewsqa_qwen35_4b_base_without_metadata_target_matrix_qwen35_4b_base_seed41.jsonl`
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

### 83. LocalNewsQA appendix model-gain table | Qwen3.5-4B Base | LocalNewsQA Explicit

- model name: `Qwen3.5-4B Base`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta +0.0257, CI [-0.1695, +0.2312], n=19465
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_4b/base/seed_41/target/localnewsqa_qwen35_4b_base_with_metadata_target_matrix_qwen35_4b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_4b/base/seed_41/target/localnewsqa_qwen35_4b_base_without_metadata_target_matrix_qwen35_4b_base_seed41.jsonl`
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

### 84. LocalNewsQA appendix model-gain table | Qwen3.5-4B Base | LocalNewsQA Exact pair

- model name: `Qwen3.5-4B Base`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +1.65, CI [+1.45, +1.85], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_4b/base/seed_41/target/localnewsqa_qwen35_4b_base_with_metadata_target_matrix_qwen35_4b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/qwen35_4b/base/seed_41/contrast/localnewsqa_qwen35_4b_base_with_metadata_contrast_matrix_qwen35_4b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_4b/base/seed_41/target/localnewsqa_qwen35_4b_base_without_metadata_target_matrix_qwen35_4b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/qwen35_4b/base/seed_41/contrast/localnewsqa_qwen35_4b_base_without_metadata_contrast_matrix_qwen35_4b_base_seed41.jsonl`
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

### 85. LocalNewsQA appendix model-gain table | Qwen3.5-4B Base | LocalNewsQA Margin switch

- model name: `Qwen3.5-4B Base`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +3.38, CI [+3.11, +3.64], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_4b/base/seed_41/target/localnewsqa_qwen35_4b_base_with_metadata_target_matrix_qwen35_4b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/qwen35_4b/base/seed_41/contrast/localnewsqa_qwen35_4b_base_with_metadata_contrast_matrix_qwen35_4b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/qwen35_4b/base/seed_41/target/localnewsqa_qwen35_4b_base_without_metadata_target_matrix_qwen35_4b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/qwen35_4b/base/seed_41/contrast/localnewsqa_qwen35_4b_base_without_metadata_contrast_matrix_qwen35_4b_base_seed41.jsonl`
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

### 86. LocalNewsQA appendix model-gain table | Qwen3.5-4B Chat | LocalNewsQA Overall

- model name: `Qwen3.5-4B Chat`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta +0.5687, CI [+0.3762, +0.7666], n=35874
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_4b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_4b_without_metadata_target.jsonl`
- fixed-example positive prompt:
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

### 87. LocalNewsQA appendix model-gain table | Qwen3.5-4B Chat | LocalNewsQA Ambiguous

- model name: `Qwen3.5-4B Chat`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +1.04, CI [+0.7251, +1.35], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_4b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_4b_without_metadata_target.jsonl`
- fixed-example positive prompt:
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

### 88. LocalNewsQA appendix model-gain table | Qwen3.5-4B Chat | LocalNewsQA Explicit

- model name: `Qwen3.5-4B Chat`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta +0.1695, CI [-0.0771, +0.4367], n=19465
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_4b_with_metadata_target.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_4b_without_metadata_target.jsonl`
- fixed-example positive prompt:
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

### 89. LocalNewsQA appendix model-gain table | Qwen3.5-4B Chat | LocalNewsQA Exact pair

- model name: `Qwen3.5-4B Chat`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +1.23, CI [+1.07, +1.40], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_4b_with_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_qwen35_4b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_4b_without_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_qwen35_4b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
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

### 90. LocalNewsQA appendix model-gain table | Qwen3.5-4B Chat | LocalNewsQA Margin switch

- model name: `Qwen3.5-4B Chat`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +2.71, CI [+2.45, +2.96], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_4b_with_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_qwen35_4b_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_qwen35_4b_without_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_qwen35_4b_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
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

### 91. LocalNewsQA appendix model-gain table | Gemma-4-E2B Base | LocalNewsQA Overall

- model name: `Gemma-4-E2B Base`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta +2.25, CI [+1.83, +2.66], n=35874
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/gemma4_e2b/base/seed_41/target/localnewsqa_gemma4_e2b_base_with_metadata_target_matrix_gemma4_e2b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/gemma4_e2b/base/seed_41/target/localnewsqa_gemma4_e2b_base_without_metadata_target_matrix_gemma4_e2b_base_seed41.jsonl`
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

### 92. LocalNewsQA appendix model-gain table | Gemma-4-E2B Base | LocalNewsQA Ambiguous

- model name: `Gemma-4-E2B Base`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +2.44, CI [+1.87, +3.03], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/gemma4_e2b/base/seed_41/target/localnewsqa_gemma4_e2b_base_with_metadata_target_matrix_gemma4_e2b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/gemma4_e2b/base/seed_41/target/localnewsqa_gemma4_e2b_base_without_metadata_target_matrix_gemma4_e2b_base_seed41.jsonl`
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

### 93. LocalNewsQA appendix model-gain table | Gemma-4-E2B Base | LocalNewsQA Explicit

- model name: `Gemma-4-E2B Base`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta +2.08, CI [+1.58, +2.61], n=19465
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/gemma4_e2b/base/seed_41/target/localnewsqa_gemma4_e2b_base_with_metadata_target_matrix_gemma4_e2b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/gemma4_e2b/base/seed_41/target/localnewsqa_gemma4_e2b_base_without_metadata_target_matrix_gemma4_e2b_base_seed41.jsonl`
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

### 94. LocalNewsQA appendix model-gain table | Gemma-4-E2B Base | LocalNewsQA Exact pair

- model name: `Gemma-4-E2B Base`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +0.5241, CI [+0.3778, +0.6705], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/gemma4_e2b/base/seed_41/target/localnewsqa_gemma4_e2b_base_with_metadata_target_matrix_gemma4_e2b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/gemma4_e2b/base/seed_41/contrast/localnewsqa_gemma4_e2b_base_with_metadata_contrast_matrix_gemma4_e2b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/gemma4_e2b/base/seed_41/target/localnewsqa_gemma4_e2b_base_without_metadata_target_matrix_gemma4_e2b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/gemma4_e2b/base/seed_41/contrast/localnewsqa_gemma4_e2b_base_without_metadata_contrast_matrix_gemma4_e2b_base_seed41.jsonl`
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

### 95. LocalNewsQA appendix model-gain table | Gemma-4-E2B Base | LocalNewsQA Margin switch

- model name: `Gemma-4-E2B Base`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +1.47, CI [+1.24, +1.71], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/gemma4_e2b/base/seed_41/target/localnewsqa_gemma4_e2b_base_with_metadata_target_matrix_gemma4_e2b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/gemma4_e2b/base/seed_41/contrast/localnewsqa_gemma4_e2b_base_with_metadata_contrast_matrix_gemma4_e2b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/gemma4_e2b/base/seed_41/target/localnewsqa_gemma4_e2b_base_without_metadata_target_matrix_gemma4_e2b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/gemma4_e2b/base/seed_41/contrast/localnewsqa_gemma4_e2b_base_without_metadata_contrast_matrix_gemma4_e2b_base_seed41.jsonl`
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

### 96. LocalNewsQA appendix model-gain table | Gemma-4-E2B-it | LocalNewsQA Overall

- model name: `Gemma-4-E2B-it`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta +1.93, CI [+1.66, +2.19], n=35874
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_gemma4_e2b_it_with_metadata_target.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_gemma4_e2b_it_without_metadata_target.jsonl`
- fixed-example positive prompt:
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

### 97. LocalNewsQA appendix model-gain table | Gemma-4-E2B-it | LocalNewsQA Ambiguous

- model name: `Gemma-4-E2B-it`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +3.49, CI [+3.06, +3.92], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_gemma4_e2b_it_with_metadata_target.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_gemma4_e2b_it_without_metadata_target.jsonl`
- fixed-example positive prompt:
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

### 98. LocalNewsQA appendix model-gain table | Gemma-4-E2B-it | LocalNewsQA Explicit

- model name: `Gemma-4-E2B-it`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta +0.6062, CI [+0.2774, +0.9196], n=19465
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_gemma4_e2b_it_with_metadata_target.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_gemma4_e2b_it_without_metadata_target.jsonl`
- fixed-example positive prompt:
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

### 99. LocalNewsQA appendix model-gain table | Gemma-4-E2B-it | LocalNewsQA Exact pair

- model name: `Gemma-4-E2B-it`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +4.03, CI [+3.75, +4.35], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_gemma4_e2b_it_with_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_gemma4_e2b_it_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_gemma4_e2b_it_without_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_gemma4_e2b_it_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
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

### 100. LocalNewsQA appendix model-gain table | Gemma-4-E2B-it | LocalNewsQA Margin switch

- model name: `Gemma-4-E2B-it`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +8.25, CI [+7.83, +8.65], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_gemma4_e2b_it_with_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_gemma4_e2b_it_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_gemma4_e2b_it_without_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_gemma4_e2b_it_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
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

### 101. LocalNewsQA appendix model-gain table | Gemma-4-E4B Base | LocalNewsQA Overall

- model name: `Gemma-4-E4B Base`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta +0.8474, CI [+0.4572, +1.22], n=35874
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/gemma4_e4b/base/seed_41/target/localnewsqa_gemma4_e4b_base_with_metadata_target_matrix_gemma4_e4b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/gemma4_e4b/base/seed_41/target/localnewsqa_gemma4_e4b_base_without_metadata_target_matrix_gemma4_e4b_base_seed41.jsonl`
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

### 102. LocalNewsQA appendix model-gain table | Gemma-4-E4B Base | LocalNewsQA Ambiguous

- model name: `Gemma-4-E4B Base`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +1.13, CI [+0.5727, +1.71], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/gemma4_e4b/base/seed_41/target/localnewsqa_gemma4_e4b_base_with_metadata_target_matrix_gemma4_e4b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/gemma4_e4b/base/seed_41/target/localnewsqa_gemma4_e4b_base_without_metadata_target_matrix_gemma4_e4b_base_seed41.jsonl`
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

### 103. LocalNewsQA appendix model-gain table | Gemma-4-E4B Base | LocalNewsQA Explicit

- model name: `Gemma-4-E4B Base`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta +0.6114, CI [+0.1180, +1.09], n=19465
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/gemma4_e4b/base/seed_41/target/localnewsqa_gemma4_e4b_base_with_metadata_target_matrix_gemma4_e4b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/gemma4_e4b/base/seed_41/target/localnewsqa_gemma4_e4b_base_without_metadata_target_matrix_gemma4_e4b_base_seed41.jsonl`
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

### 104. LocalNewsQA appendix model-gain table | Gemma-4-E4B Base | LocalNewsQA Exact pair

- model name: `Gemma-4-E4B Base`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +1.24, CI [+1.08, +1.42], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/gemma4_e4b/base/seed_41/target/localnewsqa_gemma4_e4b_base_with_metadata_target_matrix_gemma4_e4b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/gemma4_e4b/base/seed_41/contrast/localnewsqa_gemma4_e4b_base_with_metadata_contrast_matrix_gemma4_e4b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/gemma4_e4b/base/seed_41/target/localnewsqa_gemma4_e4b_base_without_metadata_target_matrix_gemma4_e4b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/gemma4_e4b/base/seed_41/contrast/localnewsqa_gemma4_e4b_base_without_metadata_contrast_matrix_gemma4_e4b_base_seed41.jsonl`
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

### 105. LocalNewsQA appendix model-gain table | Gemma-4-E4B Base | LocalNewsQA Margin switch

- model name: `Gemma-4-E4B Base`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +3.43, CI [+3.16, +3.71], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/gemma4_e4b/base/seed_41/target/localnewsqa_gemma4_e4b_base_with_metadata_target_matrix_gemma4_e4b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/gemma4_e4b/base/seed_41/contrast/localnewsqa_gemma4_e4b_base_with_metadata_contrast_matrix_gemma4_e4b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/gemma4_e4b/base/seed_41/target/localnewsqa_gemma4_e4b_base_without_metadata_target_matrix_gemma4_e4b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/gemma4_e4b/base/seed_41/contrast/localnewsqa_gemma4_e4b_base_without_metadata_contrast_matrix_gemma4_e4b_base_seed41.jsonl`
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

### 106. LocalNewsQA appendix model-gain table | Gemma-4-E4B-it | LocalNewsQA Overall

- model name: `Gemma-4-E4B-it`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta +4.52, CI [+4.23, +4.80], n=35874
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_gemma4_e4b_it_with_metadata_target.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_gemma4_e4b_it_without_metadata_target.jsonl`
- fixed-example positive prompt:
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

### 107. LocalNewsQA appendix model-gain table | Gemma-4-E4B-it | LocalNewsQA Ambiguous

- model name: `Gemma-4-E4B-it`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +7.22, CI [+6.73, +7.73], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_gemma4_e4b_it_with_metadata_target.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_gemma4_e4b_it_without_metadata_target.jsonl`
- fixed-example positive prompt:
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

### 108. LocalNewsQA appendix model-gain table | Gemma-4-E4B-it | LocalNewsQA Explicit

- model name: `Gemma-4-E4B-it`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta +2.25, CI [+1.93, +2.54], n=19465
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_gemma4_e4b_it_with_metadata_target.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_gemma4_e4b_it_without_metadata_target.jsonl`
- fixed-example positive prompt:
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

### 109. LocalNewsQA appendix model-gain table | Gemma-4-E4B-it | LocalNewsQA Exact pair

- model name: `Gemma-4-E4B-it`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +8.46, CI [+8.02, +8.90], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_gemma4_e4b_it_with_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_gemma4_e4b_it_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_gemma4_e4b_it_without_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_gemma4_e4b_it_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
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

### 110. LocalNewsQA appendix model-gain table | Gemma-4-E4B-it | LocalNewsQA Margin switch

- model name: `Gemma-4-E4B-it`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +17.09, CI [+16.50, +17.65], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_gemma4_e4b_it_with_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_gemma4_e4b_it_with_metadata_contrast.jsonl`
- negative source path(s):
  - `results/downstream_localnewsqa_external_baselines_multiseed/seed_41/localnewsqa_gemma4_e4b_it_without_metadata_target.jsonl`
  - `results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun/seed_41/localnewsqa_gemma4_e4b_it_without_metadata_contrast.jsonl`
- fixed-example positive prompt:
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

### 111. LocalNewsQA appendix model-gain table | Ministral-3-3B Base | LocalNewsQA Overall

- model name: `Ministral-3-3B Base`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta +2.16, CI [+1.85, +2.48], n=35874
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/ministral3_3b/base/seed_41/target/localnewsqa_ministral3_3b_base_with_metadata_target_matrix_ministral3_3b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/ministral3_3b/base/seed_41/target/localnewsqa_ministral3_3b_base_without_metadata_target_matrix_ministral3_3b_base_seed41.jsonl`
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

### 112. LocalNewsQA appendix model-gain table | Ministral-3-3B Base | LocalNewsQA Ambiguous

- model name: `Ministral-3-3B Base`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +5.00, CI [+4.49, +5.49], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/ministral3_3b/base/seed_41/target/localnewsqa_ministral3_3b_base_with_metadata_target_matrix_ministral3_3b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/ministral3_3b/base/seed_41/target/localnewsqa_ministral3_3b_base_without_metadata_target_matrix_ministral3_3b_base_seed41.jsonl`
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

### 113. LocalNewsQA appendix model-gain table | Ministral-3-3B Base | LocalNewsQA Explicit

- model name: `Ministral-3-3B Base`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta -0.2312, CI [-0.6114, +0.1490], n=19465
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/ministral3_3b/base/seed_41/target/localnewsqa_ministral3_3b_base_with_metadata_target_matrix_ministral3_3b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/ministral3_3b/base/seed_41/target/localnewsqa_ministral3_3b_base_without_metadata_target_matrix_ministral3_3b_base_seed41.jsonl`
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

### 114. LocalNewsQA appendix model-gain table | Ministral-3-3B Base | LocalNewsQA Exact pair

- model name: `Ministral-3-3B Base`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +5.27, CI [+4.91, +5.61], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/ministral3_3b/base/seed_41/target/localnewsqa_ministral3_3b_base_with_metadata_target_matrix_ministral3_3b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/ministral3_3b/base/seed_41/contrast/localnewsqa_ministral3_3b_base_with_metadata_contrast_matrix_ministral3_3b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/ministral3_3b/base/seed_41/target/localnewsqa_ministral3_3b_base_without_metadata_target_matrix_ministral3_3b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/ministral3_3b/base/seed_41/contrast/localnewsqa_ministral3_3b_base_without_metadata_contrast_matrix_ministral3_3b_base_seed41.jsonl`
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

### 115. LocalNewsQA appendix model-gain table | Ministral-3-3B Base | LocalNewsQA Margin switch

- model name: `Ministral-3-3B Base`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +11.98, CI [+11.51, +12.49], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/ministral3_3b/base/seed_41/target/localnewsqa_ministral3_3b_base_with_metadata_target_matrix_ministral3_3b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/ministral3_3b/base/seed_41/contrast/localnewsqa_ministral3_3b_base_with_metadata_contrast_matrix_ministral3_3b_base_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/ministral3_3b/base/seed_41/target/localnewsqa_ministral3_3b_base_without_metadata_target_matrix_ministral3_3b_base_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/ministral3_3b/base/seed_41/contrast/localnewsqa_ministral3_3b_base_without_metadata_contrast_matrix_ministral3_3b_base_seed41.jsonl`
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

### 116. LocalNewsQA appendix model-gain table | Ministral-3-3B Inst. | LocalNewsQA Overall

- model name: `Ministral-3-3B Inst.`
- dataset name: `LocalNewsQA Overall` (`localnewsqa_overall`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on all LocalNewsQA items; plus side minus minus side, paired by item.
- result in table: delta +3.05, CI [+2.78, +3.33], n=35874
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/ministral3_3b/chat/seed_41/target/localnewsqa_ministral3_3b_chat_with_metadata_target_matrix_ministral3_3b_chat_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/ministral3_3b/chat/seed_41/target/localnewsqa_ministral3_3b_chat_without_metadata_target_matrix_ministral3_3b_chat_seed41.jsonl`
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

### 117. LocalNewsQA appendix model-gain table | Ministral-3-3B Inst. | LocalNewsQA Ambiguous

- model name: `Ministral-3-3B Inst.`
- dataset name: `LocalNewsQA Ambiguous` (`localnewsqa_ambiguous`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on ambiguous LocalNewsQA items only.
- result in table: delta +5.79, CI [+5.32, +6.27], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/ministral3_3b/chat/seed_41/target/localnewsqa_ministral3_3b_chat_with_metadata_target_matrix_ministral3_3b_chat_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/ministral3_3b/chat/seed_41/target/localnewsqa_ministral3_3b_chat_without_metadata_target_matrix_ministral3_3b_chat_seed41.jsonl`
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

### 118. LocalNewsQA appendix model-gain table | Ministral-3-3B Inst. | LocalNewsQA Explicit

- model name: `Ministral-3-3B Inst.`
- dataset name: `LocalNewsQA Explicit` (`localnewsqa_explicit`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Target-locale accuracy gain on explicit LocalNewsQA items only.
- result in table: delta +0.7449, CI [+0.4263, +1.07], n=19465
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/ministral3_3b/chat/seed_41/target/localnewsqa_ministral3_3b_chat_with_metadata_target_matrix_ministral3_3b_chat_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/ministral3_3b/chat/seed_41/target/localnewsqa_ministral3_3b_chat_without_metadata_target_matrix_ministral3_3b_chat_seed41.jsonl`
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

### 119. LocalNewsQA appendix model-gain table | Ministral-3-3B Inst. | LocalNewsQA Exact pair

- model name: `Ministral-3-3B Inst.`
- dataset name: `LocalNewsQA Exact pair` (`localnewsqa_exact_pair`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the item counts when the target prompt predicts the target answer and the contrast prompt predicts the contrast answer.
- result in table: delta +6.56, CI [+6.19, +6.94], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/ministral3_3b/chat/seed_41/target/localnewsqa_ministral3_3b_chat_with_metadata_target_matrix_ministral3_3b_chat_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/ministral3_3b/chat/seed_41/contrast/localnewsqa_ministral3_3b_chat_with_metadata_contrast_matrix_ministral3_3b_chat_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/ministral3_3b/chat/seed_41/target/localnewsqa_ministral3_3b_chat_without_metadata_target_matrix_ministral3_3b_chat_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/ministral3_3b/chat/seed_41/contrast/localnewsqa_ministral3_3b_chat_without_metadata_contrast_matrix_ministral3_3b_chat_seed41.jsonl`
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

### 120. LocalNewsQA appendix model-gain table | Ministral-3-3B Inst. | LocalNewsQA Margin switch

- model name: `Ministral-3-3B Inst.`
- dataset name: `LocalNewsQA Margin switch` (`localnewsqa_margin_switch`)
- comparison: I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt).
- metric: `paired_metric_pp_gain`; Ambiguous-item target/contrast pair metric: the target-answer margin must be positive under the target prompt and the contrast-answer margin must be positive under the contrast prompt.
- result in table: delta +14.53, CI [+13.99, +15.05], n=16409
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=1.0; add_prompt_bos=False
- table rescore: alpha=None, beta=None; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/localnewsqa/ministral3_3b/chat/seed_41/target/localnewsqa_ministral3_3b_chat_with_metadata_target_matrix_ministral3_3b_chat_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/ministral3_3b/chat/seed_41/contrast/localnewsqa_ministral3_3b_chat_with_metadata_contrast_matrix_ministral3_3b_chat_seed41.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/localnewsqa/ministral3_3b/chat/seed_41/target/localnewsqa_ministral3_3b_chat_without_metadata_target_matrix_ministral3_3b_chat_seed41.jsonl`
  - `results/final_benchmark_matrix/localnewsqa/ministral3_3b/chat/seed_41/contrast/localnewsqa_ministral3_3b_chat_without_metadata_contrast_matrix_ministral3_3b_chat_seed41.jsonl`
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

### 121. External benchmark appendix model-gain table | MAPLE 1B Base | Geo

- model name: `MAPLE 1B Base`
- dataset name: `Geo` (`geomlama`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +2.67, CI [-8.00, +13.33], n=150
- positive source protocol key: `strict_country_a2_b1`
- negative source protocol key: `strict_country_a2_b1`
- positive protocol details: metadata_prompt_style=name_strict; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=2.0; null_calibration_mode=question_masked; null_calibration_beta=1.0; add_prompt_bos=True; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=2.0; null_calibration_mode=question_masked; null_calibration_beta=1.0; add_prompt_bos=True; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/external_protocol_selected_maple_seed41/geomlama/strict_country_a2_b1/maple_1b/seed_41/geomlama/geomlama_custom_tplus_eplus.jsonl`
- negative source path(s):
  - `results/external_protocol_selected_maple_seed41/geomlama/strict_country_a2_b1/maple_1b/seed_41/geomlama/geomlama_custom_tminus_eminus.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: Canada
COUNTRY_CODE: ca
CONTINENT: America

TITLE: Facts about Canada

CONTENT:
This question is specifically about factual knowledge in Canada. Choose the answer that is correct for Canada, not the answer that would fit a different country or the answer that is only more globally common.

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

### 122. External benchmark appendix model-gain table | MAPLE 1B Base | GOQA

- model name: `MAPLE 1B Base`
- dataset name: `GOQA` (`globalopinionqa`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +1.27, CI [+0.3870, +2.14], n=9818
- positive source protocol key: `letter_country_first`
- negative source protocol key: `letter_country_first`
- positive protocol details: metadata_prompt_style=country_first_strict; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_letter; null_calibration_mode=question_masked; null_calibration_beta=0.25; add_prompt_bos=True; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_letter; null_calibration_mode=question_masked; null_calibration_beta=0.25; add_prompt_bos=True; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/external_paper_table_tokenizerfix_20260504/globalopinionqa/letter_country_first/maple_1b/seed_41/globalopinionqa/globalopinionqa_custom_tplus_eplus.jsonl`
- negative source path(s):
  - `results/external_paper_table_tokenizerfix_20260504/globalopinionqa/letter_country_first/maple_1b/seed_41/globalopinionqa/globalopinionqa_custom_tminus_eminus.jsonl`
- fixed-example positive prompt:
```text
COUNTRY: Canada
COUNTRY_CODE: ca
CONTINENT: America
URL: www.globalfactcheck.org/ca

TITLE: Facts about Canada

CONTENT:
Answer this question for Canada. When multiple options could fit different countries, pick the option that is factual for Canada.

Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

For Canada, final answer:
```
- positive scoring candidates: `[' A', ' B', ' C', ' D']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Final answer:
```
- negative scoring candidates: `[' A', ' B', ' C', ' D']`

### 123. External benchmark appendix model-gain table | MAPLE 1B Base | MMLU-CS

- model name: `MAPLE 1B Base`
- dataset name: `MMLU-CS` (`globalmmlu_cs`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +0.1263, CI [-2.40, +2.65], n=792
- positive source protocol key: `figure9_1b`
- negative source protocol key: `figure9_1b`
- positive protocol details: metadata_prompt_style=code_grounded; qa_prompt_style=question; answer_cue_style=final_answer_colon; omit_option_labels=False; exact_option_text_instruction=True; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=1.25; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=final_answer_colon; omit_option_labels=False; exact_option_text_instruction=True; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=1.25; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/external_paper_table_tokenizerfix_20260504/globalmmlu_cs/figure9_1b/maple_1b/seed_41/globalmmlu_cs/globalmmlu_cs_custom_tplus_eplus.jsonl`
- negative source path(s):
  - `results/external_paper_table_tokenizerfix_20260504/globalmmlu_cs/figure9_1b/maple_1b/seed_41/globalmmlu_cs/globalmmlu_cs_custom_tminus_eminus.jsonl`
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

### 124. External benchmark appendix model-gain table | MAPLE 1B Base | NormAD

- model name: `MAPLE 1B Base`
- dataset name: `NormAD` (`normad`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +1.44, CI [+0.2659, +2.58], n=2633
- positive source protocol key: `figure9_1b`
- negative source protocol key: `figure9_1b`
- positive protocol details: metadata_prompt_style=code_grounded; qa_prompt_style=question; answer_cue_style=final_answer_colon; omit_option_labels=False; exact_option_text_instruction=True; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=1.25; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=final_answer_colon; omit_option_labels=False; exact_option_text_instruction=True; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=1.25; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/external_paper_table_tokenizerfix_20260504/normad/figure9_1b/maple_1b/seed_41/normad/normad_custom_tplus_eplus.jsonl`
- negative source path(s):
  - `results/external_paper_table_tokenizerfix_20260504/normad/figure9_1b/maple_1b/seed_41/normad/normad_custom_tminus_eminus.jsonl`
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

### 125. External benchmark appendix model-gain table | MAPLE 1B Base | BLEnD

- model name: `MAPLE 1B Base`
- dataset name: `BLEnD` (`blend`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +9.41, CI [+3.05, +15.27], n=393
- positive source protocol key: `letter_country_first`
- negative source protocol key: `letter_country_first`
- positive protocol details: metadata_prompt_style=country_first_strict; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_letter; null_calibration_mode=question_masked; null_calibration_beta=0.25; add_prompt_bos=True; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_letter; null_calibration_mode=question_masked; null_calibration_beta=0.25; add_prompt_bos=True; metadata_tag_mode=correct
- table rescore: alpha=0.0, beta=0.975; Table recomputes argmax from saved option log-likelihood sums with this alpha/beta.
- positive source path(s):
  - `results/external_paper_table_tokenizerfix_20260504/blend/letter_country_first/maple_1b/seed_41/blend/blend_custom_tplus_eplus.jsonl`
- negative source path(s):
  - `results/external_paper_table_tokenizerfix_20260504/blend/letter_country_first/maple_1b/seed_41/blend/blend_custom_tminus_eminus.jsonl`
- fixed-example positive prompt:
```text
COUNTRY: Canada
COUNTRY_CODE: ca
CONTINENT: America
URL: www.globalfactcheck.org/ca

TITLE: Facts about Canada

CONTENT:
Answer this question for Canada. When multiple options could fit different countries, pick the option that is factual for Canada.

Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

For Canada, final answer:
```
- positive scoring candidates: `[' A', ' B', ' C', ' D']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Final answer:
```
- negative scoring candidates: `[' A', ' B', ' C', ' D']`

### 126. External benchmark appendix model-gain table | MAPLE 1B Base | WVB

- model name: `MAPLE 1B Base`
- dataset name: `WVB` (`worldvaluebench`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `wvb_emd_gain`; WVB gain: reduction in Earth-mover distance, so positive means the plus side is closer.
- result in table: delta +0.0271, CI [+0.0066, +0.0481], n=333
- positive source protocol key: `letter_country_first`
- negative source protocol key: `letter_country_first`
- positive protocol details: metadata_prompt_style=country_first_strict; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_letter; null_calibration_mode=question_masked; null_calibration_beta=0.25; add_prompt_bos=True; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_letter; null_calibration_mode=question_masked; null_calibration_beta=0.25; add_prompt_bos=True; metadata_tag_mode=correct
- table rescore: alpha=0.0, beta=1.3; Table recomputes argmax from saved option log-likelihood sums with this alpha/beta.
- positive source path(s):
  - `results/external_paper_table_tokenizerfix_20260504/worldvaluebench/letter_country_first/maple_1b/seed_41/worldvaluebench/worldvaluebench_custom_tplus_eplus.jsonl`
- negative source path(s):
  - `results/external_paper_table_tokenizerfix_20260504/worldvaluebench/letter_country_first/maple_1b/seed_41/worldvaluebench/worldvaluebench_custom_tminus_eminus.jsonl`
- fixed-example positive prompt:
```text
COUNTRY: Canada
COUNTRY_CODE: ca
CONTINENT: America
URL: www.globalfactcheck.org/ca

TITLE: Facts about Canada

CONTENT:
Answer this question for Canada. When multiple options could fit different countries, pick the option that is factual for Canada.

Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

For Canada, final answer:
```
- positive scoring candidates: `[' A', ' B', ' C', ' D']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Final answer:
```
- negative scoring candidates: `[' A', ' B', ' C', ' D']`

### 127. External benchmark appendix model-gain table | MAPLE 1B Chat | Geo

- model name: `MAPLE 1B Chat`
- dataset name: `Geo` (`geomlama`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +4.00, CI [-6.00, +13.35], n=150
- positive source protocol key: `strict_country`
- negative source protocol key: `strict_country`
- positive protocol details: metadata_prompt_style=name_strict; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.25; add_prompt_bos=True; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.25; add_prompt_bos=True; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/external_protocol_selected_maple_chat_seed41/geomlama/strict_country/maple_1b_chat/seed_41/geomlama/geomlama_custom_tplus_eplus.jsonl`
- negative source path(s):
  - `results/external_protocol_selected_maple_chat_seed41/geomlama/strict_country/maple_1b_chat/seed_41/geomlama/geomlama_custom_tminus_eminus.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: Canada
COUNTRY_CODE: ca
CONTINENT: America

TITLE: Facts about Canada

CONTENT:
This question is specifically about factual knowledge in Canada. Choose the answer that is correct for Canada, not the answer that would fit a different country or the answer that is only more globally common.

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

### 128. External benchmark appendix model-gain table | MAPLE 1B Chat | GOQA

- model name: `MAPLE 1B Chat`
- dataset name: `GOQA` (`globalopinionqa`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +2.72, CI [+1.75, +3.64], n=9818
- positive source protocol key: `strict_country_a0_b15`
- negative source protocol key: `strict_country_a0_b15`
- positive protocol details: metadata_prompt_style=name_strict; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.0; null_calibration_mode=question_masked; null_calibration_beta=1.5; add_prompt_bos=True; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.0; null_calibration_mode=question_masked; null_calibration_beta=1.5; add_prompt_bos=True; metadata_tag_mode=correct
- table rescore: alpha=0.5, beta=1.5; Table recomputes argmax from saved option log-likelihood sums with this alpha/beta.
- positive source path(s):
  - `results/external_protocol_selected_maple_chat_seed41/globalopinionqa/strict_country_a0_b15/maple_1b_chat/seed_41/globalopinionqa/globalopinionqa_custom_tplus_eplus.jsonl`
- negative source path(s):
  - `results/external_protocol_selected_maple_chat_seed41/globalopinionqa/strict_country_a0_b15/maple_1b_chat/seed_41/globalopinionqa/globalopinionqa_custom_tminus_eminus.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: Canada
COUNTRY_CODE: ca
CONTINENT: America

TITLE: Facts about Canada

CONTENT:
This question is specifically about factual knowledge in Canada. Choose the answer that is correct for Canada, not the answer that would fit a different country or the answer that is only more globally common.

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

### 129. External benchmark appendix model-gain table | MAPLE 1B Chat | MMLU-CS

- model name: `MAPLE 1B Chat`
- dataset name: `MMLU-CS` (`globalmmlu_cs`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +0.2569, CI [-0.5922, +1.07], n=14015
- positive source protocol key: `figure9_1b`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=code_grounded; qa_prompt_style=question; answer_cue_style=final_answer_colon; omit_option_labels=False; exact_option_text_instruction=True; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=1.25; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=final_answer_colon; omit_option_labels=False; exact_option_text_instruction=True; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=1.25; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=0.0, beta=0.0; Table recomputes argmax from saved option log-likelihood sums with this alpha/beta.
- positive source path(s):
  - `results/external_benchmarks_maple_tuned_seed41/sft_1b_chat/seed_41/mmlu/custom_tplus_eplus/mmlu_custom_tplus_eplus.jsonl`
- negative source path(s):
  - `results/external_benchmarks_maple_tuned_seed41/sft_1b_chat/seed_41/mmlu/custom_tminus_eminus/mmlu_custom_tminus_eminus.jsonl`
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

### 130. External benchmark appendix model-gain table | MAPLE 1B Chat | NormAD

- model name: `MAPLE 1B Chat`
- dataset name: `NormAD` (`normad`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +3.80, CI [+0.8735, +6.38], n=2633
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/maple_1b/chat/seed_41/normad/normad_custom_tplus_eplus.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/maple_1b/chat/seed_41/normad/normad_custom_tminus_eminus.jsonl`
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

### 131. External benchmark appendix model-gain table | MAPLE 1B Chat | BLEnD

- model name: `MAPLE 1B Chat`
- dataset name: `BLEnD` (`blend`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +1.02, CI [-2.04, +4.07], n=393
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=1.25, beta=0.0; Table recomputes argmax from saved option log-likelihood sums with this alpha/beta.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/maple_1b/chat/seed_41/blend/blend_custom_tplus_eplus.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/maple_1b/chat/seed_41/blend/blend_custom_tminus_eminus.jsonl`
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

### 132. External benchmark appendix model-gain table | MAPLE 1B Chat | WVB

- model name: `MAPLE 1B Chat`
- dataset name: `WVB` (`worldvaluebench`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `wvb_emd_gain`; WVB gain: reduction in Earth-mover distance, so positive means the plus side is closer.
- result in table: delta +0.0141, CI [+0.0075, +0.0213], n=333
- positive source protocol key: `figure9_3b`
- negative source protocol key: `figure9_3b`
- positive protocol details: metadata_prompt_style=name_grounded; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/external_protocol_selected_maple_chat_seed41/worldvaluebench/figure9_3b/maple_1b_chat/seed_41/worldvaluebench/worldvaluebench_custom_tplus_eplus.jsonl`
- negative source path(s):
  - `results/external_protocol_selected_maple_chat_seed41/worldvaluebench/figure9_3b/maple_1b_chat/seed_41/worldvaluebench/worldvaluebench_custom_tminus_eminus.jsonl`
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

### 133. External benchmark appendix model-gain table | MAPLE 3B Base | Geo

- model name: `MAPLE 3B Base`
- dataset name: `Geo` (`geomlama`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +5.33, CI [-4.67, +14.67], n=150
- positive source protocol key: `sft_1b_lnqa`
- negative source protocol key: `sft_1b_lnqa`
- positive protocol details: metadata_prompt_style=name_plain; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.25; add_prompt_bos=True; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.25; add_prompt_bos=True; metadata_tag_mode=correct
- table rescore: alpha=0.35, beta=1.0; Table recomputes argmax from saved option log-likelihood sums with this alpha/beta.
- positive source path(s):
  - `results/external_protocol_selected_maple_seed41/geomlama/sft_1b_lnqa/maple_3b/seed_41/geomlama/geomlama_custom_tplus_eplus.jsonl`
- negative source path(s):
  - `results/external_protocol_selected_maple_seed41/geomlama/sft_1b_lnqa/maple_3b/seed_41/geomlama/geomlama_custom_tminus_eminus.jsonl`
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

### 134. External benchmark appendix model-gain table | MAPLE 3B Base | GOQA

- model name: `MAPLE 3B Base`
- dataset name: `GOQA` (`globalopinionqa`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +8.44, CI [+7.26, +9.55], n=9818
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_grounded; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True; metadata_tag_mode=correct
- table rescore: alpha=1.0, beta=1.0; Table recomputes argmax from saved option log-likelihood sums with this alpha/beta.
- positive source path(s):
  - `results/external_benchmarks_maple_tuned_seed41/raw_3b/seed_41/globalopinionqa/custom_tplus_eplus/globalopinionqa_custom_tplus_eplus.jsonl`
- negative source path(s):
  - `results/external_benchmarks_maple_tuned_seed41/raw_3b/seed_41/globalopinionqa/custom_tminus_eminus/globalopinionqa_custom_tminus_eminus.jsonl`
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

### 135. External benchmark appendix model-gain table | MAPLE 3B Base | MMLU-CS

- model name: `MAPLE 3B Base`
- dataset name: `MMLU-CS` (`globalmmlu_cs`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +3.91, CI [+0.5051, +7.32], n=792
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_plain; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.25; add_prompt_bos=False; metadata_tag_mode=available_only
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.25; add_prompt_bos=False; metadata_tag_mode=available_only
- table rescore: alpha=1.5, beta=0.5; Table recomputes argmax from saved option log-likelihood sums with this alpha/beta.
- positive source path(s):
  - `results/external_benchmarks_croq_related_maple_gpuq_probe_seed41/nameplain_qanswer_countryfinal_avg_qmask025_nobos/seed_41/globalmmlu_cs/custom_tplus_eplus/globalmmlu_cs_custom_tplus_eplus.jsonl`
- negative source path(s):
  - `results/external_benchmarks_croq_related_maple_gpuq_probe_seed41/nameplain_qanswer_countryfinal_avg_qmask025_nobos/seed_41/globalmmlu_cs/custom_tminus_eminus/globalmmlu_cs_custom_tminus_eminus.jsonl`
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

### 136. External benchmark appendix model-gain table | MAPLE 3B Base | NormAD

- model name: `MAPLE 3B Base`
- dataset name: `NormAD` (`normad`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +5.05, CI [+2.70, +7.37], n=2633
- positive source protocol key: `figure9_3b`
- negative source protocol key: `figure9_3b`
- positive protocol details: metadata_prompt_style=name_grounded; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/external_paper_table_tokenizerfix_20260504/normad/figure9_3b/maple_3b/seed_41/normad/normad_custom_tplus_eplus.jsonl`
- negative source path(s):
  - `results/external_paper_table_tokenizerfix_20260504/normad/figure9_3b/maple_3b/seed_41/normad/normad_custom_tminus_eminus.jsonl`
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

### 137. External benchmark appendix model-gain table | MAPLE 3B Base | BLEnD

- model name: `MAPLE 3B Base`
- dataset name: `BLEnD` (`blend`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +13.49, CI [+7.12, +19.59], n=393
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_grounded; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True; metadata_tag_mode=correct
- table rescore: alpha=1.0, beta=1.0; Table recomputes argmax from saved option log-likelihood sums with this alpha/beta.
- positive source path(s):
  - `results/external_benchmarks_croq_related_seed41/raw_3b/seed_41/blend/custom_tplus_eplus/blend_custom_tplus_eplus.jsonl`
- negative source path(s):
  - `results/external_benchmarks_croq_related_seed41/raw_3b/seed_41/blend/custom_tminus_eminus/blend_custom_tminus_eminus.jsonl`
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

### 138. External benchmark appendix model-gain table | MAPLE 3B Base | WVB

- model name: `MAPLE 3B Base`
- dataset name: `WVB` (`worldvaluebench`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `wvb_emd_gain`; WVB gain: reduction in Earth-mover distance, so positive means the plus side is closer.
- result in table: delta +0.0049, CI [-0.0014, +0.0108], n=333
- positive source protocol key: `letter_legacy`
- negative source protocol key: `letter_legacy`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=answer_colon; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_letter; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=answer_colon; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_letter; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/external_paper_table_tokenizerfix_20260504/worldvaluebench/letter_legacy/maple_3b/seed_41/worldvaluebench/worldvaluebench_custom_tplus_eplus.jsonl`
- negative source path(s):
  - `results/external_paper_table_tokenizerfix_20260504/worldvaluebench/letter_legacy/maple_3b/seed_41/worldvaluebench/worldvaluebench_custom_tminus_eminus.jsonl`
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

Answer with only the correct option letter (A, B, C, or D).

Answer:
```
- positive scoring candidates: `[' A', ' B', ' C', ' D']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Answer with only the correct option letter (A, B, C, or D).

Answer:
```
- negative scoring candidates: `[' A', ' B', ' C', ' D']`

### 139. External benchmark appendix model-gain table | MAPLE 3B Chat | Geo

- model name: `MAPLE 3B Chat`
- dataset name: `Geo` (`geomlama`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +4.67, CI [-4.67, +13.33], n=150
- positive source protocol key: `strict_country_a2_b1`
- negative source protocol key: `strict_country_a2_b1`
- positive protocol details: metadata_prompt_style=name_strict; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=2.0; null_calibration_mode=question_masked; null_calibration_beta=1.0; add_prompt_bos=True; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=2.0; null_calibration_mode=question_masked; null_calibration_beta=1.0; add_prompt_bos=True; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/external_protocol_selected_maple_chat_seed41/geomlama/strict_country_a2_b1/maple_3b_chat/seed_41/geomlama/geomlama_custom_tplus_eplus.jsonl`
- negative source path(s):
  - `results/external_protocol_selected_maple_chat_seed41/geomlama/strict_country_a2_b1/maple_3b_chat/seed_41/geomlama/geomlama_custom_tminus_eminus.jsonl`
- fixed-example positive prompt:
```text
URL: www.globalfactcheck.org/ca
COUNTRY: Canada
COUNTRY_CODE: ca
CONTINENT: America

TITLE: Facts about Canada

CONTENT:
This question is specifically about factual knowledge in Canada. Choose the answer that is correct for Canada, not the answer that would fit a different country or the answer that is only more globally common.

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

### 140. External benchmark appendix model-gain table | MAPLE 3B Chat | GOQA

- model name: `MAPLE 3B Chat`
- dataset name: `GOQA` (`globalopinionqa`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +3.42, CI [+2.55, +4.35], n=9818
- positive source protocol key: `legacy_a025`
- negative source protocol key: `legacy_a025`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/external_protocol_selected_maple_chat_seed41/globalopinionqa/legacy_a025/maple_3b_chat/seed_41/globalopinionqa/globalopinionqa_custom_tplus_eplus.jsonl`
- negative source path(s):
  - `results/external_protocol_selected_maple_chat_seed41/globalopinionqa/legacy_a025/maple_3b_chat/seed_41/globalopinionqa/globalopinionqa_custom_tminus_eminus.jsonl`
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

### 141. External benchmark appendix model-gain table | MAPLE 3B Chat | MMLU-CS

- model name: `MAPLE 3B Chat`
- dataset name: `MMLU-CS` (`globalmmlu_cs`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +0.6778, CI [-0.1429, +1.52], n=14015
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=name_grounded; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True; metadata_tag_mode=correct
- table rescore: alpha=0.2, beta=0.25; Table recomputes argmax from saved option log-likelihood sums with this alpha/beta.
- positive source path(s):
  - `results/external_benchmarks_maple_tuned_seed41/sft_3b_best3b_chat/seed_41/mmlu/custom_tplus_eplus/mmlu_custom_tplus_eplus.jsonl`
- negative source path(s):
  - `results/external_benchmarks_maple_tuned_seed41/sft_3b_best3b_chat/seed_41/mmlu/custom_tminus_eminus/mmlu_custom_tminus_eminus.jsonl`
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

### 142. External benchmark appendix model-gain table | MAPLE 3B Chat | NormAD

- model name: `MAPLE 3B Chat`
- dataset name: `NormAD` (`normad`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +2.43, CI [+0.1899, +4.63], n=2633
- positive source protocol key: `letter_name_grounded`
- negative source protocol key: `letter_name_grounded`
- positive protocol details: metadata_prompt_style=name_grounded; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_letter; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=final_answer_colon; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_letter; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True; metadata_tag_mode=correct
- table rescore: alpha=0.0, beta=0.725; Table recomputes argmax from saved option log-likelihood sums with this alpha/beta.
- positive source path(s):
  - `results/external_protocol_selected_maple_chat_seed41/normad/letter_name_grounded/maple_3b_chat/seed_41/normad/normad_custom_tplus_eplus.jsonl`
- negative source path(s):
  - `results/external_protocol_selected_maple_chat_seed41/normad/letter_name_grounded/maple_3b_chat/seed_41/normad/normad_custom_tminus_eminus.jsonl`
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
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Final answer:
```
- positive scoring candidates: `[' A', ' B', ' C', ' D']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Final answer:
```
- negative scoring candidates: `[' A', ' B', ' C', ' D']`

### 143. External benchmark appendix model-gain table | MAPLE 3B Chat | BLEnD

- model name: `MAPLE 3B Chat`
- dataset name: `BLEnD` (`blend`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +6.11, CI [+0.7634, +11.70], n=393
- positive source protocol key: `name_grounded_the`
- negative source protocol key: `name_grounded_the`
- positive protocol details: metadata_prompt_style=name_grounded; qa_prompt_style=question_answer; answer_cue_style=the_correct_answer_is; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=the_correct_answer_is; omit_option_labels=True; exact_option_text_instruction=False; mcq_scoring=option_text_avg; option_loglikelihood_length_norm_alpha=0.25; null_calibration_mode=question_masked; null_calibration_beta=0.5; add_prompt_bos=True; metadata_tag_mode=correct
- table rescore: alpha=0.25, beta=0.5; Table recomputes argmax from saved option log-likelihood sums with this alpha/beta.
- positive source path(s):
  - `results/external_protocol_selected_maple_chat_seed41/blend/name_grounded_the/maple_3b_chat/seed_41/blend/blend_custom_tplus_eplus.jsonl`
- negative source path(s):
  - `results/external_protocol_selected_maple_chat_seed41/blend/name_grounded_the/maple_3b_chat/seed_41/blend/blend_custom_tminus_eminus.jsonl`
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

The correct answer is
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

The correct answer is
```
- negative scoring candidates: `[' Canadian Broadcasting Corporation', ' British Broadcasting Corporation', ' Australian Broadcasting Corporation', ' National Public Radio']`

### 144. External benchmark appendix model-gain table | MAPLE 3B Chat | WVB

- model name: `MAPLE 3B Chat`
- dataset name: `WVB` (`worldvaluebench`)
- comparison: T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without).
- metric: `wvb_emd_gain`; WVB gain: reduction in Earth-mover distance, so positive means the plus side is closer.
- result in table: delta +0.0215, CI [+0.0049, +0.0399], n=333
- positive source protocol key: `letter_country_first`
- negative source protocol key: `letter_country_first`
- positive protocol details: metadata_prompt_style=country_first_strict; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_letter; null_calibration_mode=question_masked; null_calibration_beta=0.25; add_prompt_bos=True; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question_answer; answer_cue_style=country_final_answer_colon; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_letter; null_calibration_mode=question_masked; null_calibration_beta=0.25; add_prompt_bos=True; metadata_tag_mode=correct
- table rescore: alpha=0.0, beta=0.6; Table recomputes argmax from saved option log-likelihood sums with this alpha/beta.
- positive source path(s):
  - `results/external_protocol_selected_maple_chat_seed41/worldvaluebench/letter_country_first/maple_3b_chat/seed_41/worldvaluebench/worldvaluebench_custom_tplus_eplus.jsonl`
- negative source path(s):
  - `results/external_protocol_selected_maple_chat_seed41/worldvaluebench/letter_country_first/maple_3b_chat/seed_41/worldvaluebench/worldvaluebench_custom_tminus_eminus.jsonl`
- fixed-example positive prompt:
```text
COUNTRY: Canada
COUNTRY_CODE: ca
CONTINENT: America
URL: www.globalfactcheck.org/ca

TITLE: Facts about Canada

CONTENT:
Answer this question for Canada. When multiple options could fit different countries, pick the option that is factual for Canada.

Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

For Canada, final answer:
```
- positive scoring candidates: `[' A', ' B', ' C', ' D']`
- fixed-example negative prompt:
```text
CONTENT:
Question: Which institution is the national public broadcaster for this locale?

Options:
A: Canadian Broadcasting Corporation
B: British Broadcasting Corporation
C: Australian Broadcasting Corporation
D: National Public Radio

Final answer:
```
- negative scoring candidates: `[' A', ' B', ' C', ' D']`

### 145. External benchmark appendix model-gain table | LLaMA-3.2-1B Base | Geo

- model name: `LLaMA-3.2-1B Base`
- dataset name: `Geo` (`geomlama`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -2.67, CI [-5.33, -0.6667], n=150
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_1b/base/seed_41/geomlama/llama3_chat_with_metadata/geomlama_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_1b/base/seed_41/geomlama/llama3_chat_without_metadata/geomlama_llama3_chat_without_metadata.jsonl`
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

### 146. External benchmark appendix model-gain table | LLaMA-3.2-1B Base | GOQA

- model name: `LLaMA-3.2-1B Base`
- dataset name: `GOQA` (`globalopinionqa`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -0.1222, CI [-0.5296, +0.2852], n=9818
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_1b/base/seed_41/globalopinionqa/llama3_chat_with_metadata/globalopinionqa_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_1b/base/seed_41/globalopinionqa/llama3_chat_without_metadata/globalopinionqa_llama3_chat_without_metadata.jsonl`
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

### 147. External benchmark appendix model-gain table | LLaMA-3.2-1B Base | MMLU-CS

- model name: `LLaMA-3.2-1B Base`
- dataset name: `MMLU-CS` (`globalmmlu_cs`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +2.02, CI [+0.5051, +3.54], n=792
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_1b/base/seed_41/globalmmlu_cs/llama3_chat_with_metadata/globalmmlu_cs_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_1b/base/seed_41/globalmmlu_cs/llama3_chat_without_metadata/globalmmlu_cs_llama3_chat_without_metadata.jsonl`
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

### 148. External benchmark appendix model-gain table | LLaMA-3.2-1B Base | NormAD

- model name: `LLaMA-3.2-1B Base`
- dataset name: `NormAD` (`normad`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +1.41, CI [-0.5317, +3.46], n=2633
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_1b/base/seed_41/normad/llama3_chat_with_metadata/normad_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_1b/base/seed_41/normad/llama3_chat_without_metadata/normad_llama3_chat_without_metadata.jsonl`
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

### 149. External benchmark appendix model-gain table | LLaMA-3.2-1B Base | BLEnD

- model name: `LLaMA-3.2-1B Base`
- dataset name: `BLEnD` (`blend`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +0.0000, CI [-2.04, +2.04], n=393
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_1b/base/seed_41/blend/llama3_chat_with_metadata/blend_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_1b/base/seed_41/blend/llama3_chat_without_metadata/blend_llama3_chat_without_metadata.jsonl`
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

### 150. External benchmark appendix model-gain table | LLaMA-3.2-1B Base | WVB

- model name: `LLaMA-3.2-1B Base`
- dataset name: `WVB` (`worldvaluebench`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `wvb_emd_gain`; WVB gain: reduction in Earth-mover distance, so positive means the plus side is closer.
- result in table: delta +0.0169, CI [-0.0199, +0.0526], n=333
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_1b/base/seed_41/worldvaluebench/llama3_chat_with_metadata/worldvaluebench_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_1b/base/seed_41/worldvaluebench/llama3_chat_without_metadata/worldvaluebench_llama3_chat_without_metadata.jsonl`
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

### 151. External benchmark appendix model-gain table | LLaMA-3.2-1B Inst. | Geo

- model name: `LLaMA-3.2-1B Inst.`
- dataset name: `Geo` (`geomlama`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +5.33, CI [-2.67, +12.67], n=150
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_1b/chat/seed_41/geomlama/geomlama_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_1b/chat/seed_41/geomlama/geomlama_llama3_chat_without_metadata.jsonl`
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

### 152. External benchmark appendix model-gain table | LLaMA-3.2-1B Inst. | GOQA

- model name: `LLaMA-3.2-1B Inst.`
- dataset name: `GOQA` (`globalopinionqa`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -0.2444, CI [-0.6111, +0.1222], n=9818
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_1b/chat/seed_41/globalopinionqa/globalopinionqa_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_1b/chat/seed_41/globalopinionqa/globalopinionqa_llama3_chat_without_metadata.jsonl`
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

### 153. External benchmark appendix model-gain table | LLaMA-3.2-1B Inst. | MMLU-CS

- model name: `LLaMA-3.2-1B Inst.`
- dataset name: `MMLU-CS` (`globalmmlu_cs`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -1.01, CI [-2.53, +0.5051], n=792
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_1b/chat/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_1b/chat/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_without_metadata.jsonl`
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

### 154. External benchmark appendix model-gain table | LLaMA-3.2-1B Inst. | NormAD

- model name: `LLaMA-3.2-1B Inst.`
- dataset name: `NormAD` (`normad`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +0.0000, CI [+0.0000, +0.0000], n=2633
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_1b/chat/seed_41/normad/normad_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_1b/chat/seed_41/normad/normad_llama3_chat_without_metadata.jsonl`
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

### 155. External benchmark appendix model-gain table | LLaMA-3.2-1B Inst. | BLEnD

- model name: `LLaMA-3.2-1B Inst.`
- dataset name: `BLEnD` (`blend`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -1.27, CI [-4.07, +1.27], n=393
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_1b/chat/seed_41/blend/blend_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_1b/chat/seed_41/blend/blend_llama3_chat_without_metadata.jsonl`
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

### 156. External benchmark appendix model-gain table | LLaMA-3.2-1B Inst. | WVB

- model name: `LLaMA-3.2-1B Inst.`
- dataset name: `WVB` (`worldvaluebench`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `wvb_emd_gain`; WVB gain: reduction in Earth-mover distance, so positive means the plus side is closer.
- result in table: delta +0.0308, CI [+0.0174, +0.0446], n=333
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_1b/chat/seed_41/worldvaluebench/worldvaluebench_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_1b/chat/seed_41/worldvaluebench/worldvaluebench_llama3_chat_without_metadata.jsonl`
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

### 157. External benchmark appendix model-gain table | LLaMA-3.2-3B Base | Geo

- model name: `LLaMA-3.2-3B Base`
- dataset name: `Geo` (`geomlama`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -5.33, CI [-14.00, +2.67], n=150
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_3b/base/seed_41/geomlama/llama3_chat_with_metadata/geomlama_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_3b/base/seed_41/geomlama/llama3_chat_without_metadata/geomlama_llama3_chat_without_metadata.jsonl`
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

### 158. External benchmark appendix model-gain table | LLaMA-3.2-3B Base | GOQA

- model name: `LLaMA-3.2-3B Base`
- dataset name: `GOQA` (`globalopinionqa`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +0.3972, CI [+0.1118, +0.6929], n=9818
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_3b/base/seed_41/globalopinionqa/llama3_chat_with_metadata/globalopinionqa_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_3b/base/seed_41/globalopinionqa/llama3_chat_without_metadata/globalopinionqa_llama3_chat_without_metadata.jsonl`
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

### 159. External benchmark appendix model-gain table | LLaMA-3.2-3B Base | MMLU-CS

- model name: `LLaMA-3.2-3B Base`
- dataset name: `MMLU-CS` (`globalmmlu_cs`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +0.7576, CI [-0.5051, +2.02], n=792
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_3b/base/seed_41/globalmmlu_cs/llama3_chat_with_metadata/globalmmlu_cs_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_3b/base/seed_41/globalmmlu_cs/llama3_chat_without_metadata/globalmmlu_cs_llama3_chat_without_metadata.jsonl`
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

### 160. External benchmark appendix model-gain table | LLaMA-3.2-3B Base | NormAD

- model name: `LLaMA-3.2-3B Base`
- dataset name: `NormAD` (`normad`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +1.90, CI [-0.1139, +3.87], n=2633
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_3b/base/seed_41/normad/llama3_chat_with_metadata/normad_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_3b/base/seed_41/normad/llama3_chat_without_metadata/normad_llama3_chat_without_metadata.jsonl`
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

### 161. External benchmark appendix model-gain table | LLaMA-3.2-3B Base | BLEnD

- model name: `LLaMA-3.2-3B Base`
- dataset name: `BLEnD` (`blend`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +1.53, CI [-0.2545, +3.31], n=393
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_3b/base/seed_41/blend/llama3_chat_with_metadata/blend_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_3b/base/seed_41/blend/llama3_chat_without_metadata/blend_llama3_chat_without_metadata.jsonl`
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

### 162. External benchmark appendix model-gain table | LLaMA-3.2-3B Base | WVB

- model name: `LLaMA-3.2-3B Base`
- dataset name: `WVB` (`worldvaluebench`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `wvb_emd_gain`; WVB gain: reduction in Earth-mover distance, so positive means the plus side is closer.
- result in table: delta -0.0256, CI [-0.0373, -0.0156], n=333
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_3b/base/seed_41/worldvaluebench/llama3_chat_with_metadata/worldvaluebench_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_3b/base/seed_41/worldvaluebench/llama3_chat_without_metadata/worldvaluebench_llama3_chat_without_metadata.jsonl`
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

### 163. External benchmark appendix model-gain table | LLaMA-3.2-3B Inst. | Geo

- model name: `LLaMA-3.2-3B Inst.`
- dataset name: `Geo` (`geomlama`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -4.67, CI [-12.67, +4.00], n=150
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_3b/chat/seed_41/geomlama/llama3_chat_with_metadata/geomlama_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_3b/chat/seed_41/geomlama/llama3_chat_without_metadata/geomlama_llama3_chat_without_metadata.jsonl`
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

### 164. External benchmark appendix model-gain table | LLaMA-3.2-3B Inst. | GOQA

- model name: `LLaMA-3.2-3B Inst.`
- dataset name: `GOQA` (`globalopinionqa`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -0.9574, CI [-1.36, -0.5602], n=9818
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_3b/chat/seed_41/globalopinionqa/llama3_chat_with_metadata/globalopinionqa_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_3b/chat/seed_41/globalopinionqa/llama3_chat_without_metadata/globalopinionqa_llama3_chat_without_metadata.jsonl`
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

### 165. External benchmark appendix model-gain table | LLaMA-3.2-3B Inst. | MMLU-CS

- model name: `LLaMA-3.2-3B Inst.`
- dataset name: `MMLU-CS` (`globalmmlu_cs`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -2.90, CI [-4.80, -1.01], n=792
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_3b/chat/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_3b/chat/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_without_metadata.jsonl`
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

### 166. External benchmark appendix model-gain table | LLaMA-3.2-3B Inst. | NormAD

- model name: `LLaMA-3.2-3B Inst.`
- dataset name: `NormAD` (`normad`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -0.5317, CI [-1.60, +0.5317], n=2633
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_3b/chat/seed_41/normad/normad_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_3b/chat/seed_41/normad/normad_llama3_chat_without_metadata.jsonl`
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

### 167. External benchmark appendix model-gain table | LLaMA-3.2-3B Inst. | BLEnD

- model name: `LLaMA-3.2-3B Inst.`
- dataset name: `BLEnD` (`blend`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -9.16, CI [-12.98, -5.85], n=393
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_3b/chat/seed_41/blend/blend_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_3b/chat/seed_41/blend/blend_llama3_chat_without_metadata.jsonl`
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

### 168. External benchmark appendix model-gain table | LLaMA-3.2-3B Inst. | WVB

- model name: `LLaMA-3.2-3B Inst.`
- dataset name: `WVB` (`worldvaluebench`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `wvb_emd_gain`; WVB gain: reduction in Earth-mover distance, so positive means the plus side is closer.
- result in table: delta -0.0955, CI [-0.1258, -0.0652], n=333
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_3b/chat/seed_41/worldvaluebench/worldvaluebench_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/llama32_3b/chat/seed_41/worldvaluebench/worldvaluebench_llama3_chat_without_metadata.jsonl`
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

### 169. External benchmark appendix model-gain table | Qwen2.5-1.5B Base | Geo

- model name: `Qwen2.5-1.5B Base`
- dataset name: `Geo` (`geomlama`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -10.67, CI [-16.00, -5.33], n=150
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_1p5b/base/seed_41/geomlama/llama3_chat_with_metadata/geomlama_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_1p5b/base/seed_41/geomlama/llama3_chat_without_metadata/geomlama_llama3_chat_without_metadata.jsonl`
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

### 170. External benchmark appendix model-gain table | Qwen2.5-1.5B Base | GOQA

- model name: `Qwen2.5-1.5B Base`
- dataset name: `GOQA` (`globalopinionqa`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -2.10, CI [-2.68, -1.53], n=9818
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_1p5b/base/seed_41/globalopinionqa/llama3_chat_with_metadata/globalopinionqa_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_1p5b/base/seed_41/globalopinionqa/llama3_chat_without_metadata/globalopinionqa_llama3_chat_without_metadata.jsonl`
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

### 171. External benchmark appendix model-gain table | Qwen2.5-1.5B Base | MMLU-CS

- model name: `Qwen2.5-1.5B Base`
- dataset name: `MMLU-CS` (`globalmmlu_cs`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +0.7576, CI [-1.01, +2.40], n=792
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_1p5b/base/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_1p5b/base/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_without_metadata.jsonl`
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

### 172. External benchmark appendix model-gain table | Qwen2.5-1.5B Base | NormAD

- model name: `Qwen2.5-1.5B Base`
- dataset name: `NormAD` (`normad`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +4.98, CI [+1.75, +8.05], n=2633
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_1p5b/base/seed_41/normad/normad_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_1p5b/base/seed_41/normad/normad_llama3_chat_without_metadata.jsonl`
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

### 173. External benchmark appendix model-gain table | Qwen2.5-1.5B Base | BLEnD

- model name: `Qwen2.5-1.5B Base`
- dataset name: `BLEnD` (`blend`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -0.7634, CI [-2.80, +1.27], n=393
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_1p5b/base/seed_41/blend/blend_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_1p5b/base/seed_41/blend/blend_llama3_chat_without_metadata.jsonl`
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

### 174. External benchmark appendix model-gain table | Qwen2.5-1.5B Base | WVB

- model name: `Qwen2.5-1.5B Base`
- dataset name: `WVB` (`worldvaluebench`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `wvb_emd_gain`; WVB gain: reduction in Earth-mover distance, so positive means the plus side is closer.
- result in table: delta +0.0062, CI [+0.0000, +0.0138], n=333
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_1p5b/base/seed_41/worldvaluebench/worldvaluebench_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_1p5b/base/seed_41/worldvaluebench/worldvaluebench_llama3_chat_without_metadata.jsonl`
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

### 175. External benchmark appendix model-gain table | Qwen2.5-1.5B Inst. | Geo

- model name: `Qwen2.5-1.5B Inst.`
- dataset name: `Geo` (`geomlama`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -8.67, CI [-14.67, -2.67], n=150
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_1p5b/chat/seed_41/geomlama/llama3_chat_with_metadata/geomlama_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_1p5b/chat/seed_41/geomlama/llama3_chat_without_metadata/geomlama_llama3_chat_without_metadata.jsonl`
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

### 176. External benchmark appendix model-gain table | Qwen2.5-1.5B Inst. | GOQA

- model name: `Qwen2.5-1.5B Inst.`
- dataset name: `GOQA` (`globalopinionqa`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -0.8250, CI [-1.24, -0.3972], n=9818
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_1p5b/chat/seed_41/globalopinionqa/llama3_chat_with_metadata/globalopinionqa_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_1p5b/chat/seed_41/globalopinionqa/llama3_chat_without_metadata/globalopinionqa_llama3_chat_without_metadata.jsonl`
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

### 177. External benchmark appendix model-gain table | Qwen2.5-1.5B Inst. | MMLU-CS

- model name: `Qwen2.5-1.5B Inst.`
- dataset name: `MMLU-CS` (`globalmmlu_cs`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -0.2525, CI [-1.89, +1.39], n=792
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_1p5b/chat/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_1p5b/chat/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_without_metadata.jsonl`
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

### 178. External benchmark appendix model-gain table | Qwen2.5-1.5B Inst. | NormAD

- model name: `Qwen2.5-1.5B Inst.`
- dataset name: `NormAD` (`normad`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -2.70, CI [-4.06, -1.44], n=2633
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_1p5b/chat/seed_41/normad/normad_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_1p5b/chat/seed_41/normad/normad_llama3_chat_without_metadata.jsonl`
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

### 179. External benchmark appendix model-gain table | Qwen2.5-1.5B Inst. | BLEnD

- model name: `Qwen2.5-1.5B Inst.`
- dataset name: `BLEnD` (`blend`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -1.53, CI [-4.33, +1.27], n=393
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_1p5b/chat/seed_41/blend/blend_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_1p5b/chat/seed_41/blend/blend_llama3_chat_without_metadata.jsonl`
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

### 180. External benchmark appendix model-gain table | Qwen2.5-1.5B Inst. | WVB

- model name: `Qwen2.5-1.5B Inst.`
- dataset name: `WVB` (`worldvaluebench`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `wvb_emd_gain`; WVB gain: reduction in Earth-mover distance, so positive means the plus side is closer.
- result in table: delta -0.0222, CI [-0.0372, -0.0069], n=333
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_1p5b/chat/seed_41/worldvaluebench/worldvaluebench_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_1p5b/chat/seed_41/worldvaluebench/worldvaluebench_llama3_chat_without_metadata.jsonl`
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

### 181. External benchmark appendix model-gain table | Qwen2.5-3B Base | Geo

- model name: `Qwen2.5-3B Base`
- dataset name: `Geo` (`geomlama`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +0.0000, CI [-5.33, +6.00], n=150
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_3b/base/seed_41/geomlama/llama3_chat_with_metadata/geomlama_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_3b/base/seed_41/geomlama/llama3_chat_without_metadata/geomlama_llama3_chat_without_metadata.jsonl`
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

### 182. External benchmark appendix model-gain table | Qwen2.5-3B Base | GOQA

- model name: `Qwen2.5-3B Base`
- dataset name: `GOQA` (`globalopinionqa`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -0.4380, CI [-0.7945, -0.0713], n=9818
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_3b/base/seed_41/globalopinionqa/llama3_chat_with_metadata/globalopinionqa_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_3b/base/seed_41/globalopinionqa/llama3_chat_without_metadata/globalopinionqa_llama3_chat_without_metadata.jsonl`
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

### 183. External benchmark appendix model-gain table | Qwen2.5-3B Base | MMLU-CS

- model name: `Qwen2.5-3B Base`
- dataset name: `MMLU-CS` (`globalmmlu_cs`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -0.2525, CI [-2.02, +1.52], n=792
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_3b/base/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_3b/base/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_without_metadata.jsonl`
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

### 184. External benchmark appendix model-gain table | Qwen2.5-3B Base | NormAD

- model name: `Qwen2.5-3B Base`
- dataset name: `NormAD` (`normad`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +9.57, CI [+7.82, +11.39], n=2633
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_3b/base/seed_41/normad/normad_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_3b/base/seed_41/normad/normad_llama3_chat_without_metadata.jsonl`
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

### 185. External benchmark appendix model-gain table | Qwen2.5-3B Base | BLEnD

- model name: `Qwen2.5-3B Base`
- dataset name: `BLEnD` (`blend`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -1.78, CI [-4.07, +0.5089], n=393
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_3b/base/seed_41/blend/blend_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_3b/base/seed_41/blend/blend_llama3_chat_without_metadata.jsonl`
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

### 186. External benchmark appendix model-gain table | Qwen2.5-3B Base | WVB

- model name: `Qwen2.5-3B Base`
- dataset name: `WVB` (`worldvaluebench`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `wvb_emd_gain`; WVB gain: reduction in Earth-mover distance, so positive means the plus side is closer.
- result in table: delta -0.0223, CI [-0.0375, -0.0060], n=333
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_3b/base/seed_41/worldvaluebench/worldvaluebench_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_3b/base/seed_41/worldvaluebench/worldvaluebench_llama3_chat_without_metadata.jsonl`
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

### 187. External benchmark appendix model-gain table | Qwen2.5-3B Inst. | Geo

- model name: `Qwen2.5-3B Inst.`
- dataset name: `Geo` (`geomlama`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -2.67, CI [-9.33, +4.00], n=150
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_3b/chat/seed_41/geomlama/llama3_chat_with_metadata/geomlama_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_3b/chat/seed_41/geomlama/llama3_chat_without_metadata/geomlama_llama3_chat_without_metadata.jsonl`
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

### 188. External benchmark appendix model-gain table | Qwen2.5-3B Inst. | GOQA

- model name: `Qwen2.5-3B Inst.`
- dataset name: `GOQA` (`globalopinionqa`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -0.2444, CI [-0.7028, +0.1833], n=9818
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_3b/chat/seed_41/globalopinionqa/llama3_chat_with_metadata/globalopinionqa_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_3b/chat/seed_41/globalopinionqa/llama3_chat_without_metadata/globalopinionqa_llama3_chat_without_metadata.jsonl`
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

### 189. External benchmark appendix model-gain table | Qwen2.5-3B Inst. | MMLU-CS

- model name: `Qwen2.5-3B Inst.`
- dataset name: `MMLU-CS` (`globalmmlu_cs`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +0.0000, CI [-1.77, +1.77], n=792
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_3b/chat/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_3b/chat/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_without_metadata.jsonl`
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

### 190. External benchmark appendix model-gain table | Qwen2.5-3B Inst. | NormAD

- model name: `Qwen2.5-3B Inst.`
- dataset name: `NormAD` (`normad`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -0.0760, CI [-1.14, +0.9875], n=2633
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_3b/chat/seed_41/normad/normad_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_3b/chat/seed_41/normad/normad_llama3_chat_without_metadata.jsonl`
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

### 191. External benchmark appendix model-gain table | Qwen2.5-3B Inst. | BLEnD

- model name: `Qwen2.5-3B Inst.`
- dataset name: `BLEnD` (`blend`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +1.27, CI [-1.53, +4.07], n=393
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_3b/chat/seed_41/blend/blend_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_3b/chat/seed_41/blend/blend_llama3_chat_without_metadata.jsonl`
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

### 192. External benchmark appendix model-gain table | Qwen2.5-3B Inst. | WVB

- model name: `Qwen2.5-3B Inst.`
- dataset name: `WVB` (`worldvaluebench`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `wvb_emd_gain`; WVB gain: reduction in Earth-mover distance, so positive means the plus side is closer.
- result in table: delta +0.0092, CI [-0.0040, +0.0235], n=333
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_3b/chat/seed_41/worldvaluebench/worldvaluebench_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen25_3b/chat/seed_41/worldvaluebench/worldvaluebench_llama3_chat_without_metadata.jsonl`
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

### 193. External benchmark appendix model-gain table | Qwen3.5-0.8B Base | Geo

- model name: `Qwen3.5-0.8B Base`
- dataset name: `Geo` (`geomlama`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -1.33, CI [-5.33, +2.67], n=150
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_0p8b/base/seed_41/geomlama/llama3_chat_with_metadata/geomlama_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_0p8b/base/seed_41/geomlama/llama3_chat_without_metadata/geomlama_llama3_chat_without_metadata.jsonl`
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

### 194. External benchmark appendix model-gain table | Qwen3.5-0.8B Base | GOQA

- model name: `Qwen3.5-0.8B Base`
- dataset name: `GOQA` (`globalopinionqa`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -0.0204, CI [-0.3870, +0.3361], n=9818
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_0p8b/base/seed_41/globalopinionqa/llama3_chat_with_metadata/globalopinionqa_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_0p8b/base/seed_41/globalopinionqa/llama3_chat_without_metadata/globalopinionqa_llama3_chat_without_metadata.jsonl`
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

### 195. External benchmark appendix model-gain table | Qwen3.5-0.8B Base | MMLU-CS

- model name: `Qwen3.5-0.8B Base`
- dataset name: `MMLU-CS` (`globalmmlu_cs`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +0.7576, CI [-0.7576, +2.27], n=792
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_0p8b/base/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_0p8b/base/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_without_metadata.jsonl`
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

### 196. External benchmark appendix model-gain table | Qwen3.5-0.8B Base | NormAD

- model name: `Qwen3.5-0.8B Base`
- dataset name: `NormAD` (`normad`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -0.4937, CI [-1.44, +0.4937], n=2633
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_0p8b/base/seed_41/normad/normad_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_0p8b/base/seed_41/normad/normad_llama3_chat_without_metadata.jsonl`
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

### 197. External benchmark appendix model-gain table | Qwen3.5-0.8B Base | BLEnD

- model name: `Qwen3.5-0.8B Base`
- dataset name: `BLEnD` (`blend`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -0.7634, CI [-2.80, +1.27], n=393
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_0p8b/base/seed_41/blend/blend_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_0p8b/base/seed_41/blend/blend_llama3_chat_without_metadata.jsonl`
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

### 198. External benchmark appendix model-gain table | Qwen3.5-0.8B Base | WVB

- model name: `Qwen3.5-0.8B Base`
- dataset name: `WVB` (`worldvaluebench`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `wvb_emd_gain`; WVB gain: reduction in Earth-mover distance, so positive means the plus side is closer.
- result in table: delta +0.0013, CI [-0.0101, +0.0127], n=333
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_0p8b/base/seed_41/worldvaluebench/worldvaluebench_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_0p8b/base/seed_41/worldvaluebench/worldvaluebench_llama3_chat_without_metadata.jsonl`
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

### 199. External benchmark appendix model-gain table | Qwen3.5-0.8B Chat | Geo

- model name: `Qwen3.5-0.8B Chat`
- dataset name: `Geo` (`geomlama`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -2.67, CI [-6.67, +0.6667], n=150
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_0p8b/chat/seed_41/geomlama/llama3_chat_with_metadata/geomlama_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_0p8b/chat/seed_41/geomlama/llama3_chat_without_metadata/geomlama_llama3_chat_without_metadata.jsonl`
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

### 200. External benchmark appendix model-gain table | Qwen3.5-0.8B Chat | GOQA

- model name: `Qwen3.5-0.8B Chat`
- dataset name: `GOQA` (`globalopinionqa`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -0.2241, CI [-0.5500, +0.1120], n=9818
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_0p8b/chat/seed_41/globalopinionqa/llama3_chat_with_metadata/globalopinionqa_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_0p8b/chat/seed_41/globalopinionqa/llama3_chat_without_metadata/globalopinionqa_llama3_chat_without_metadata.jsonl`
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

### 201. External benchmark appendix model-gain table | Qwen3.5-0.8B Chat | MMLU-CS

- model name: `Qwen3.5-0.8B Chat`
- dataset name: `MMLU-CS` (`globalmmlu_cs`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +1.01, CI [-0.1263, +2.27], n=792
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_0p8b/chat/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_0p8b/chat/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_without_metadata.jsonl`
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

### 202. External benchmark appendix model-gain table | Qwen3.5-0.8B Chat | NormAD

- model name: `Qwen3.5-0.8B Chat`
- dataset name: `NormAD` (`normad`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -2.13, CI [-3.08, -1.29], n=2633
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_0p8b/chat/seed_41/normad/normad_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_0p8b/chat/seed_41/normad/normad_llama3_chat_without_metadata.jsonl`
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

### 203. External benchmark appendix model-gain table | Qwen3.5-0.8B Chat | BLEnD

- model name: `Qwen3.5-0.8B Chat`
- dataset name: `BLEnD` (`blend`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -2.04, CI [-4.33, +0.2545], n=393
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_0p8b/chat/seed_41/blend/blend_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_0p8b/chat/seed_41/blend/blend_llama3_chat_without_metadata.jsonl`
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

### 204. External benchmark appendix model-gain table | Qwen3.5-0.8B Chat | WVB

- model name: `Qwen3.5-0.8B Chat`
- dataset name: `WVB` (`worldvaluebench`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `wvb_emd_gain`; WVB gain: reduction in Earth-mover distance, so positive means the plus side is closer.
- result in table: delta +0.0399, CI [+0.0180, +0.0644], n=333
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_0p8b/chat/seed_41/worldvaluebench/worldvaluebench_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_0p8b/chat/seed_41/worldvaluebench/worldvaluebench_llama3_chat_without_metadata.jsonl`
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

### 205. External benchmark appendix model-gain table | Qwen3.5-2B Base | Geo

- model name: `Qwen3.5-2B Base`
- dataset name: `Geo` (`geomlama`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -3.33, CI [-6.67, +0.0000], n=150
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_2b/base/seed_41/geomlama/llama3_chat_with_metadata/geomlama_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_2b/base/seed_41/geomlama/llama3_chat_without_metadata/geomlama_llama3_chat_without_metadata.jsonl`
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

### 206. External benchmark appendix model-gain table | Qwen3.5-2B Base | GOQA

- model name: `Qwen3.5-2B Base`
- dataset name: `GOQA` (`globalopinionqa`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +0.2241, CI [-0.1222, +0.5602], n=9818
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_2b/base/seed_41/globalopinionqa/llama3_chat_with_metadata/globalopinionqa_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_2b/base/seed_41/globalopinionqa/llama3_chat_without_metadata/globalopinionqa_llama3_chat_without_metadata.jsonl`
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

### 207. External benchmark appendix model-gain table | Qwen3.5-2B Base | MMLU-CS

- model name: `Qwen3.5-2B Base`
- dataset name: `MMLU-CS` (`globalmmlu_cs`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +0.6313, CI [-0.7576, +2.15], n=792
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_2b/base/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_2b/base/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_without_metadata.jsonl`
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

### 208. External benchmark appendix model-gain table | Qwen3.5-2B Base | NormAD

- model name: `Qwen3.5-2B Base`
- dataset name: `NormAD` (`normad`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -2.51, CI [-3.57, -1.48], n=2633
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_2b/base/seed_41/normad/normad_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_2b/base/seed_41/normad/normad_llama3_chat_without_metadata.jsonl`
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

### 209. External benchmark appendix model-gain table | Qwen3.5-2B Base | BLEnD

- model name: `Qwen3.5-2B Base`
- dataset name: `BLEnD` (`blend`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +1.02, CI [-1.27, +3.56], n=393
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_2b/base/seed_41/blend/blend_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_2b/base/seed_41/blend/blend_llama3_chat_without_metadata.jsonl`
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

### 210. External benchmark appendix model-gain table | Qwen3.5-2B Base | WVB

- model name: `Qwen3.5-2B Base`
- dataset name: `WVB` (`worldvaluebench`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `wvb_emd_gain`; WVB gain: reduction in Earth-mover distance, so positive means the plus side is closer.
- result in table: delta +0.0098, CI [-0.0023, +0.0216], n=333
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_2b/base/seed_41/worldvaluebench/worldvaluebench_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_2b/base/seed_41/worldvaluebench/worldvaluebench_llama3_chat_without_metadata.jsonl`
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

### 211. External benchmark appendix model-gain table | Qwen3.5-2B Chat | Geo

- model name: `Qwen3.5-2B Chat`
- dataset name: `Geo` (`geomlama`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +1.33, CI [-3.33, +6.67], n=150
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_2b/chat/seed_41/geomlama/llama3_chat_with_metadata/geomlama_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_2b/chat/seed_41/geomlama/llama3_chat_without_metadata/geomlama_llama3_chat_without_metadata.jsonl`
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

### 212. External benchmark appendix model-gain table | Qwen3.5-2B Chat | GOQA

- model name: `Qwen3.5-2B Chat`
- dataset name: `GOQA` (`globalopinionqa`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +0.0102, CI [-0.4074, +0.4077], n=9818
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_2b/chat/seed_41/globalopinionqa/llama3_chat_with_metadata/globalopinionqa_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_2b/chat/seed_41/globalopinionqa/llama3_chat_without_metadata/globalopinionqa_llama3_chat_without_metadata.jsonl`
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

### 213. External benchmark appendix model-gain table | Qwen3.5-2B Chat | MMLU-CS

- model name: `Qwen3.5-2B Chat`
- dataset name: `MMLU-CS` (`globalmmlu_cs`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +0.7576, CI [-0.8870, +2.40], n=792
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_2b/chat/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_2b/chat/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_without_metadata.jsonl`
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

### 214. External benchmark appendix model-gain table | Qwen3.5-2B Chat | NormAD

- model name: `Qwen3.5-2B Chat`
- dataset name: `NormAD` (`normad`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -3.99, CI [-4.90, -3.11], n=2633
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_2b/chat/seed_41/normad/normad_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_2b/chat/seed_41/normad/normad_llama3_chat_without_metadata.jsonl`
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

### 215. External benchmark appendix model-gain table | Qwen3.5-2B Chat | BLEnD

- model name: `Qwen3.5-2B Chat`
- dataset name: `BLEnD` (`blend`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +0.2545, CI [-2.29, +2.54], n=393
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_2b/chat/seed_41/blend/blend_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_2b/chat/seed_41/blend/blend_llama3_chat_without_metadata.jsonl`
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

### 216. External benchmark appendix model-gain table | Qwen3.5-2B Chat | WVB

- model name: `Qwen3.5-2B Chat`
- dataset name: `WVB` (`worldvaluebench`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `wvb_emd_gain`; WVB gain: reduction in Earth-mover distance, so positive means the plus side is closer.
- result in table: delta -0.0103, CI [-0.0268, +0.0067], n=333
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_2b/chat/seed_41/worldvaluebench/worldvaluebench_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_2b/chat/seed_41/worldvaluebench/worldvaluebench_llama3_chat_without_metadata.jsonl`
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

### 217. External benchmark appendix model-gain table | Qwen3.5-4B Base | Geo

- model name: `Qwen3.5-4B Base`
- dataset name: `Geo` (`geomlama`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +2.00, CI [-4.67, +8.67], n=150
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_4b/base/seed_41/geomlama/llama3_chat_with_metadata/geomlama_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_4b/base/seed_41/geomlama/llama3_chat_without_metadata/geomlama_llama3_chat_without_metadata.jsonl`
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

### 218. External benchmark appendix model-gain table | Qwen3.5-4B Base | GOQA

- model name: `Qwen3.5-4B Base`
- dataset name: `GOQA` (`globalopinionqa`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -0.2343, CI [-0.4892, +0.0204], n=9818
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_4b/base/seed_41/globalopinionqa/llama3_chat_with_metadata/globalopinionqa_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_4b/base/seed_41/globalopinionqa/llama3_chat_without_metadata/globalopinionqa_llama3_chat_without_metadata.jsonl`
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

### 219. External benchmark appendix model-gain table | Qwen3.5-4B Base | MMLU-CS

- model name: `Qwen3.5-4B Base`
- dataset name: `MMLU-CS` (`globalmmlu_cs`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +0.5051, CI [-0.5051, +1.52], n=792
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_4b/base/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_4b/base/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_without_metadata.jsonl`
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

### 220. External benchmark appendix model-gain table | Qwen3.5-4B Base | NormAD

- model name: `Qwen3.5-4B Base`
- dataset name: `NormAD` (`normad`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +0.4178, CI [+0.1519, +0.7216], n=2633
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_4b/base/seed_41/normad/normad_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_4b/base/seed_41/normad/normad_llama3_chat_without_metadata.jsonl`
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

### 221. External benchmark appendix model-gain table | Qwen3.5-4B Base | BLEnD

- model name: `Qwen3.5-4B Base`
- dataset name: `BLEnD` (`blend`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +0.5089, CI [+0.0000, +1.27], n=393
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_4b/base/seed_41/blend/blend_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_4b/base/seed_41/blend/blend_llama3_chat_without_metadata.jsonl`
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

### 222. External benchmark appendix model-gain table | Qwen3.5-4B Base | WVB

- model name: `Qwen3.5-4B Base`
- dataset name: `WVB` (`worldvaluebench`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `wvb_emd_gain`; WVB gain: reduction in Earth-mover distance, so positive means the plus side is closer.
- result in table: delta +0.0000, CI [+0.0000, +0.0000], n=333
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_4b/base/seed_41/worldvaluebench/worldvaluebench_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_4b/base/seed_41/worldvaluebench/worldvaluebench_llama3_chat_without_metadata.jsonl`
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

### 223. External benchmark appendix model-gain table | Qwen3.5-4B Chat | Geo

- model name: `Qwen3.5-4B Chat`
- dataset name: `Geo` (`geomlama`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -3.33, CI [-6.67, -0.6667], n=150
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_4b/chat/seed_41/geomlama/llama3_chat_with_metadata/geomlama_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_4b/chat/seed_41/geomlama/llama3_chat_without_metadata/geomlama_llama3_chat_without_metadata.jsonl`
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

### 224. External benchmark appendix model-gain table | Qwen3.5-4B Chat | GOQA

- model name: `Qwen3.5-4B Chat`
- dataset name: `GOQA` (`globalopinionqa`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -0.1732, CI [-0.4787, +0.1324], n=9818
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_4b/chat/seed_41/globalopinionqa/llama3_chat_with_metadata/globalopinionqa_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_4b/chat/seed_41/globalopinionqa/llama3_chat_without_metadata/globalopinionqa_llama3_chat_without_metadata.jsonl`
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

### 225. External benchmark appendix model-gain table | Qwen3.5-4B Chat | MMLU-CS

- model name: `Qwen3.5-4B Chat`
- dataset name: `MMLU-CS` (`globalmmlu_cs`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -0.3788, CI [-1.52, +0.6313], n=792
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_4b/chat/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_4b/chat/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_without_metadata.jsonl`
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

### 226. External benchmark appendix model-gain table | Qwen3.5-4B Chat | NormAD

- model name: `Qwen3.5-4B Chat`
- dataset name: `NormAD` (`normad`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +0.0000, CI [+0.0000, +0.0000], n=2633
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_4b/chat/seed_41/normad/normad_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_4b/chat/seed_41/normad/normad_llama3_chat_without_metadata.jsonl`
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

### 227. External benchmark appendix model-gain table | Qwen3.5-4B Chat | BLEnD

- model name: `Qwen3.5-4B Chat`
- dataset name: `BLEnD` (`blend`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -1.53, CI [-2.80, -0.5089], n=393
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_4b/chat/seed_41/blend/blend_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_4b/chat/seed_41/blend/blend_llama3_chat_without_metadata.jsonl`
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

### 228. External benchmark appendix model-gain table | Qwen3.5-4B Chat | WVB

- model name: `Qwen3.5-4B Chat`
- dataset name: `WVB` (`worldvaluebench`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `wvb_emd_gain`; WVB gain: reduction in Earth-mover distance, so positive means the plus side is closer.
- result in table: delta +0.0000, CI [+0.0000, +0.0000], n=333
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_4b/chat/seed_41/worldvaluebench/worldvaluebench_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/qwen35_4b/chat/seed_41/worldvaluebench/worldvaluebench_llama3_chat_without_metadata.jsonl`
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

### 229. External benchmark appendix model-gain table | Gemma-4-E2B Base | Geo

- model name: `Gemma-4-E2B Base`
- dataset name: `Geo` (`geomlama`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +2.67, CI [-1.33, +6.67], n=150
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e2b/base/seed_41/geomlama/llama3_chat_with_metadata/geomlama_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e2b/base/seed_41/geomlama/llama3_chat_without_metadata/geomlama_llama3_chat_without_metadata.jsonl`
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

### 230. External benchmark appendix model-gain table | Gemma-4-E2B Base | GOQA

- model name: `Gemma-4-E2B Base`
- dataset name: `GOQA` (`globalopinionqa`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +0.2546, CI [-0.0611, +0.5602], n=9818
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e2b/base/seed_41/globalopinionqa/llama3_chat_with_metadata/globalopinionqa_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e2b/base/seed_41/globalopinionqa/llama3_chat_without_metadata/globalopinionqa_llama3_chat_without_metadata.jsonl`
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

### 231. External benchmark appendix model-gain table | Gemma-4-E2B Base | MMLU-CS

- model name: `Gemma-4-E2B Base`
- dataset name: `MMLU-CS` (`globalmmlu_cs`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +0.2525, CI [-1.26, +1.77], n=792
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e2b/base/seed_41/globalmmlu_cs/llama3_chat_with_metadata/globalmmlu_cs_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e2b/base/seed_41/globalmmlu_cs/llama3_chat_without_metadata/globalmmlu_cs_llama3_chat_without_metadata.jsonl`
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

### 232. External benchmark appendix model-gain table | Gemma-4-E2B Base | NormAD

- model name: `Gemma-4-E2B Base`
- dataset name: `NormAD` (`normad`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +6.11, CI [+4.71, +7.52], n=2633
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e2b/base/seed_41/normad/llama3_chat_with_metadata/normad_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e2b/base/seed_41/normad/llama3_chat_without_metadata/normad_llama3_chat_without_metadata.jsonl`
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

### 233. External benchmark appendix model-gain table | Gemma-4-E2B Base | BLEnD

- model name: `Gemma-4-E2B Base`
- dataset name: `BLEnD` (`blend`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -0.2545, CI [-2.04, +1.53], n=393
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e2b/base/seed_41/blend/llama3_chat_with_metadata/blend_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e2b/base/seed_41/blend/llama3_chat_without_metadata/blend_llama3_chat_without_metadata.jsonl`
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

### 234. External benchmark appendix model-gain table | Gemma-4-E2B Base | WVB

- model name: `Gemma-4-E2B Base`
- dataset name: `WVB` (`worldvaluebench`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `wvb_emd_gain`; WVB gain: reduction in Earth-mover distance, so positive means the plus side is closer.
- result in table: delta -0.0010, CI [-0.0029, +0.0000], n=333
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e2b/base/seed_41/worldvaluebench/llama3_chat_with_metadata/worldvaluebench_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e2b/base/seed_41/worldvaluebench/llama3_chat_without_metadata/worldvaluebench_llama3_chat_without_metadata.jsonl`
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

### 235. External benchmark appendix model-gain table | Gemma-4-E2B-it | Geo

- model name: `Gemma-4-E2B-it`
- dataset name: `Geo` (`geomlama`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +2.00, CI [-6.00, +10.00], n=150
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e2b/chat/seed_41/geomlama/llama3_chat_with_metadata/geomlama_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e2b/chat/seed_41/geomlama/llama3_chat_without_metadata/geomlama_llama3_chat_without_metadata.jsonl`
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

### 236. External benchmark appendix model-gain table | Gemma-4-E2B-it | GOQA

- model name: `Gemma-4-E2B-it`
- dataset name: `GOQA` (`globalopinionqa`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -0.0509, CI [-0.4278, +0.3463], n=9818
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e2b/chat/seed_41/globalopinionqa/llama3_chat_with_metadata/globalopinionqa_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e2b/chat/seed_41/globalopinionqa/llama3_chat_without_metadata/globalopinionqa_llama3_chat_without_metadata.jsonl`
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

### 237. External benchmark appendix model-gain table | Gemma-4-E2B-it | MMLU-CS

- model name: `Gemma-4-E2B-it`
- dataset name: `MMLU-CS` (`globalmmlu_cs`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +0.5051, CI [-1.52, +2.53], n=792
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e2b/chat/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e2b/chat/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_without_metadata.jsonl`
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

### 238. External benchmark appendix model-gain table | Gemma-4-E2B-it | NormAD

- model name: `Gemma-4-E2B-it`
- dataset name: `NormAD` (`normad`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +1.10, CI [+0.0000, +2.20], n=2633
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e2b/chat/seed_41/normad/normad_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e2b/chat/seed_41/normad/normad_llama3_chat_without_metadata.jsonl`
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

### 239. External benchmark appendix model-gain table | Gemma-4-E2B-it | BLEnD

- model name: `Gemma-4-E2B-it`
- dataset name: `BLEnD` (`blend`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -0.5089, CI [-2.80, +1.78], n=393
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e2b/chat/seed_41/blend/blend_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e2b/chat/seed_41/blend/blend_llama3_chat_without_metadata.jsonl`
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

### 240. External benchmark appendix model-gain table | Gemma-4-E2B-it | WVB

- model name: `Gemma-4-E2B-it`
- dataset name: `WVB` (`worldvaluebench`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `wvb_emd_gain`; WVB gain: reduction in Earth-mover distance, so positive means the plus side is closer.
- result in table: delta +0.0291, CI [+0.0092, +0.0478], n=333
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e2b/chat/seed_41/worldvaluebench/worldvaluebench_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e2b/chat/seed_41/worldvaluebench/worldvaluebench_llama3_chat_without_metadata.jsonl`
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

### 241. External benchmark appendix model-gain table | Gemma-4-E4B Base | Geo

- model name: `Gemma-4-E4B Base`
- dataset name: `Geo` (`geomlama`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +5.33, CI [-2.67, +12.67], n=150
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e4b/base/seed_41/geomlama/llama3_chat_with_metadata/geomlama_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e4b/base/seed_41/geomlama/llama3_chat_without_metadata/geomlama_llama3_chat_without_metadata.jsonl`
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

### 242. External benchmark appendix model-gain table | Gemma-4-E4B Base | GOQA

- model name: `Gemma-4-E4B Base`
- dataset name: `GOQA` (`globalopinionqa`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -0.5500, CI [-0.9472, -0.1627], n=9818
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e4b/base/seed_41/globalopinionqa/llama3_chat_with_metadata/globalopinionqa_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e4b/base/seed_41/globalopinionqa/llama3_chat_without_metadata/globalopinionqa_llama3_chat_without_metadata.jsonl`
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

### 243. External benchmark appendix model-gain table | Gemma-4-E4B Base | MMLU-CS

- model name: `Gemma-4-E4B Base`
- dataset name: `MMLU-CS` (`globalmmlu_cs`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +2.90, CI [+1.39, +4.55], n=792
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e4b/base/seed_41/globalmmlu_cs/llama3_chat_with_metadata/globalmmlu_cs_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e4b/base/seed_41/globalmmlu_cs/llama3_chat_without_metadata/globalmmlu_cs_llama3_chat_without_metadata.jsonl`
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

### 244. External benchmark appendix model-gain table | Gemma-4-E4B Base | NormAD

- model name: `Gemma-4-E4B Base`
- dataset name: `NormAD` (`normad`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +1.71, CI [+0.9106, +2.47], n=2633
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e4b/base/seed_41/normad/llama3_chat_with_metadata/normad_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e4b/base/seed_41/normad/llama3_chat_without_metadata/normad_llama3_chat_without_metadata.jsonl`
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

### 245. External benchmark appendix model-gain table | Gemma-4-E4B Base | BLEnD

- model name: `Gemma-4-E4B Base`
- dataset name: `BLEnD` (`blend`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +5.09, CI [+2.29, +8.14], n=393
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e4b/base/seed_41/blend/llama3_chat_with_metadata/blend_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e4b/base/seed_41/blend/llama3_chat_without_metadata/blend_llama3_chat_without_metadata.jsonl`
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

### 246. External benchmark appendix model-gain table | Gemma-4-E4B Base | WVB

- model name: `Gemma-4-E4B Base`
- dataset name: `WVB` (`worldvaluebench`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `wvb_emd_gain`; WVB gain: reduction in Earth-mover distance, so positive means the plus side is closer.
- result in table: delta +0.0040, CI [-0.0037, +0.0118], n=333
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e4b/base/seed_41/worldvaluebench/llama3_chat_with_metadata/worldvaluebench_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e4b/base/seed_41/worldvaluebench/llama3_chat_without_metadata/worldvaluebench_llama3_chat_without_metadata.jsonl`
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

### 247. External benchmark appendix model-gain table | Gemma-4-E4B-it | Geo

- model name: `Gemma-4-E4B-it`
- dataset name: `Geo` (`geomlama`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +1.33, CI [-3.33, +6.02], n=150
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e4b/chat/seed_41/geomlama/llama3_chat_with_metadata/geomlama_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e4b/chat/seed_41/geomlama/llama3_chat_without_metadata/geomlama_llama3_chat_without_metadata.jsonl`
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

### 248. External benchmark appendix model-gain table | Gemma-4-E4B-it | GOQA

- model name: `Gemma-4-E4B-it`
- dataset name: `GOQA` (`globalopinionqa`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -0.9880, CI [-1.38, -0.5908], n=9818
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e4b/chat/seed_41/globalopinionqa/llama3_chat_with_metadata/globalopinionqa_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e4b/chat/seed_41/globalopinionqa/llama3_chat_without_metadata/globalopinionqa_llama3_chat_without_metadata.jsonl`
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

### 249. External benchmark appendix model-gain table | Gemma-4-E4B-it | MMLU-CS

- model name: `Gemma-4-E4B-it`
- dataset name: `MMLU-CS` (`globalmmlu_cs`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +2.65, CI [+0.7576, +4.42], n=792
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e4b/chat/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e4b/chat/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_without_metadata.jsonl`
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

### 250. External benchmark appendix model-gain table | Gemma-4-E4B-it | NormAD

- model name: `Gemma-4-E4B-it`
- dataset name: `NormAD` (`normad`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +2.51, CI [+1.18, +3.84], n=2633
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e4b/chat/seed_41/normad/normad_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e4b/chat/seed_41/normad/normad_llama3_chat_without_metadata.jsonl`
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

### 251. External benchmark appendix model-gain table | Gemma-4-E4B-it | BLEnD

- model name: `Gemma-4-E4B-it`
- dataset name: `BLEnD` (`blend`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -1.53, CI [-3.82, +0.7634], n=393
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e4b/chat/seed_41/blend/blend_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e4b/chat/seed_41/blend/blend_llama3_chat_without_metadata.jsonl`
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

### 252. External benchmark appendix model-gain table | Gemma-4-E4B-it | WVB

- model name: `Gemma-4-E4B-it`
- dataset name: `WVB` (`worldvaluebench`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `wvb_emd_gain`; WVB gain: reduction in Earth-mover distance, so positive means the plus side is closer.
- result in table: delta -0.0091, CI [-0.0243, +0.0066], n=333
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e4b/chat/seed_41/worldvaluebench/worldvaluebench_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/gemma4_e4b/chat/seed_41/worldvaluebench/worldvaluebench_llama3_chat_without_metadata.jsonl`
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

### 253. External benchmark appendix model-gain table | Ministral-3-3B Base | Geo

- model name: `Ministral-3-3B Base`
- dataset name: `Geo` (`geomlama`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -3.33, CI [-8.67, +1.33], n=150
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/ministral3_3b/base/seed_41/geomlama/llama3_chat_with_metadata/geomlama_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/ministral3_3b/base/seed_41/geomlama/llama3_chat_without_metadata/geomlama_llama3_chat_without_metadata.jsonl`
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

### 254. External benchmark appendix model-gain table | Ministral-3-3B Base | GOQA

- model name: `Ministral-3-3B Base`
- dataset name: `GOQA` (`globalopinionqa`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -0.3361, CI [-0.6417, -0.0201], n=9818
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/ministral3_3b/base/seed_41/globalopinionqa/globalopinionqa_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/ministral3_3b/base/seed_41/globalopinionqa/globalopinionqa_llama3_chat_without_metadata.jsonl`
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

### 255. External benchmark appendix model-gain table | Ministral-3-3B Base | MMLU-CS

- model name: `Ministral-3-3B Base`
- dataset name: `MMLU-CS` (`globalmmlu_cs`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +0.0000, CI [-1.26, +1.26], n=792
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/ministral3_3b/base/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/ministral3_3b/base/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_without_metadata.jsonl`
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

### 256. External benchmark appendix model-gain table | Ministral-3-3B Base | NormAD

- model name: `Ministral-3-3B Base`
- dataset name: `NormAD` (`normad`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -2.05, CI [-3.08, -1.03], n=2633
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/ministral3_3b/base/seed_41/normad/normad_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/ministral3_3b/base/seed_41/normad/normad_llama3_chat_without_metadata.jsonl`
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

### 257. External benchmark appendix model-gain table | Ministral-3-3B Base | BLEnD

- model name: `Ministral-3-3B Base`
- dataset name: `BLEnD` (`blend`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -0.5089, CI [-3.05, +2.04], n=393
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/ministral3_3b/base/seed_41/blend/blend_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/ministral3_3b/base/seed_41/blend/blend_llama3_chat_without_metadata.jsonl`
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

### 258. External benchmark appendix model-gain table | Ministral-3-3B Base | WVB

- model name: `Ministral-3-3B Base`
- dataset name: `WVB` (`worldvaluebench`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `wvb_emd_gain`; WVB gain: reduction in Earth-mover distance, so positive means the plus side is closer.
- result in table: delta -0.0364, CI [-0.0586, -0.0130], n=333
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/ministral3_3b/base/seed_41/worldvaluebench/worldvaluebench_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/ministral3_3b/base/seed_41/worldvaluebench/worldvaluebench_llama3_chat_without_metadata.jsonl`
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

### 259. External benchmark appendix model-gain table | Ministral-3-3B Inst. | Geo

- model name: `Ministral-3-3B Inst.`
- dataset name: `Geo` (`geomlama`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -12.67, CI [-18.67, -7.33], n=150
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/ministral3_3b/chat/seed_41/geomlama/llama3_chat_with_metadata/geomlama_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/ministral3_3b/chat/seed_41/geomlama/llama3_chat_without_metadata/geomlama_llama3_chat_without_metadata.jsonl`
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

### 260. External benchmark appendix model-gain table | Ministral-3-3B Inst. | GOQA

- model name: `Ministral-3-3B Inst.`
- dataset name: `GOQA` (`globalopinionqa`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta +0.0713, CI [-0.2954, +0.4482], n=9818
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/ministral3_3b/chat/seed_41/globalopinionqa/globalopinionqa_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/ministral3_3b/chat/seed_41/globalopinionqa/globalopinionqa_llama3_chat_without_metadata.jsonl`
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

### 261. External benchmark appendix model-gain table | Ministral-3-3B Inst. | MMLU-CS

- model name: `Ministral-3-3B Inst.`
- dataset name: `MMLU-CS` (`globalmmlu_cs`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -2.65, CI [-4.29, -1.14], n=792
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=available_only
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/ministral3_3b/chat/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/ministral3_3b/chat/seed_41/globalmmlu_cs/globalmmlu_cs_llama3_chat_without_metadata.jsonl`
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

### 262. External benchmark appendix model-gain table | Ministral-3-3B Inst. | NormAD

- model name: `Ministral-3-3B Inst.`
- dataset name: `NormAD` (`normad`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -4.18, CI [-5.13, -3.27], n=2633
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/ministral3_3b/chat/seed_41/normad/normad_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/ministral3_3b/chat/seed_41/normad/normad_llama3_chat_without_metadata.jsonl`
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

### 263. External benchmark appendix model-gain table | Ministral-3-3B Inst. | BLEnD

- model name: `Ministral-3-3B Inst.`
- dataset name: `BLEnD` (`blend`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `accuracy_pp_gain`; Accuracy percentage-point gain.
- result in table: delta -0.7634, CI [-2.54, +1.02], n=393
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/ministral3_3b/chat/seed_41/blend/blend_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/ministral3_3b/chat/seed_41/blend/blend_llama3_chat_without_metadata.jsonl`
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

### 264. External benchmark appendix model-gain table | Ministral-3-3B Inst. | WVB

- model name: `Ministral-3-3B Inst.`
- dataset name: `WVB` (`worldvaluebench`)
- comparison: I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt).
- metric: `wvb_emd_gain`; WVB gain: reduction in Earth-mover distance, so positive means the plus side is closer.
- result in table: delta -0.0151, CI [-0.0276, -0.0030], n=333
- positive source protocol key: `source_jsonl`
- negative source protocol key: `source_jsonl`
- positive protocol details: metadata_prompt_style=legacy_code; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- negative protocol details: metadata_prompt_style=none; qa_prompt_style=question; answer_cue_style=none; omit_option_labels=False; exact_option_text_instruction=False; mcq_scoring=option_text_avg; null_calibration_mode=none; null_calibration_beta=0.0; add_prompt_bos=False; metadata_tag_mode=correct
- table rescore: alpha=None, beta=0.0; Table uses source JSONL selected scores / processed predictions.
- positive source path(s):
  - `results/final_benchmark_matrix/closed_form/ministral3_3b/chat/seed_41/worldvaluebench/worldvaluebench_llama3_chat_with_metadata.jsonl`
- negative source path(s):
  - `results/final_benchmark_matrix/closed_form/ministral3_3b/chat/seed_41/worldvaluebench/worldvaluebench_llama3_chat_without_metadata.jsonl`
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
