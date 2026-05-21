#!/bin/bash
set -euo pipefail

MANIFEST=/path/to/logs/slurm_logs/figure9_3b_ambfix_seed41_$(date +%Y%m%d_%H%M%S).tsv
mkdir -p /path/to/logs/slurm_logs
: > "$MANIFEST"

MODEL_WITH=/path/to/metacul/models/combined_with_metadata_3b
MODEL_WITHOUT=/path/to/metacul/models/combined_without_metadata_3b
OUT_ROOT=/path/to/metacul/results/downstream_localnewsqa_pretrained_figure9_3b_ambfix_seed41

configs=(
  "correctis_nometa_a250_b025|METADATA_PROMPT_STYLE=name_grounded,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=the_correct_answer_is,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=2.5,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=question_masked_no_metadata,NULL_CALIBRATION_BETA=0.25"
  "correctis_qmask_a025_b125|METADATA_PROMPT_STYLE=name_grounded,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=the_correct_answer_is,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=1.25"
  "codeg_nolabel_qfinal_a225|METADATA_PROMPT_STYLE=code_grounded,QA_PROMPT_STYLE=question,ANSWER_CUE_STYLE=final_answer_colon,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=1,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=2.25,ADD_PROMPT_BOS=0,NULL_CALIBRATION_MODE=none,NULL_CALIBRATION_BETA=0.0"
  "namegrounded_countrycorrectis_a025_b050|METADATA_PROMPT_STYLE=name_grounded,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_the_correct_answer_is,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.5"
  "namestrict_countrycorrectis_a025_b050|METADATA_PROMPT_STYLE=name_strict,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=country_the_correct_answer_is,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=0.5"
)

for spec in "${configs[@]}"; do
  slug=${spec%%|*}
  exports=${spec#*|}
  outdir="${OUT_ROOT}/${slug}/seed_41"
  mkdir -p "$outdir"

  for train_meta in with_metadata without_metadata; do
    if [[ "$train_meta" == "with_metadata" ]]; then
      train_tag=with_metadata
      train_short=tplus
    else
      train_tag=without_metadata
      train_short=tminus
    fi

    for eval_meta in with_metadata without_metadata; do
      if [[ "$eval_meta" == "with_metadata" ]]; then
        eval_short=eplus
      else
        eval_short=eminus
      fi

      jid=$(sbatch --parsable \
        --export=ALL,MODEL_TYPE=custom,TRAIN_META_TAG=${train_tag},EVAL_META_TAG=${eval_meta},META_TAG=${eval_meta},OUT_DIR=${outdir},RUN_TAG=${slug}_3b_${train_short}_${eval_short}_seed41,CUSTOM_MODEL_PATH_WITH=${MODEL_WITH},CUSTOM_MODEL_PATH_WITHOUT=${MODEL_WITHOUT},SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED=41,ANSWER_FORMAT=option,SAMPLE_SEED=41,NULL_QUESTION_TEXT=N/A,${exports} \
        /path/to/metacul/slurm/pretrained_localnewsqa_eval_single.slurm)
      printf '%s\t%s\t%s\t%s\t%s\n' "$jid" "$slug" "$train_tag" "$eval_meta" "$exports" >> "$MANIFEST"
    done
  done
done

echo "$MANIFEST"
