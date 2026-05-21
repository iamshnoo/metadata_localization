#!/bin/bash
set -euo pipefail

MANIFEST=/path/to/logs/slurm_logs/figure9_3b_correctis_qmask_multiseed_followup_$(date +%Y%m%d_%H%M%S).tsv
mkdir -p /path/to/logs/slurm_logs
: > "$MANIFEST"

MODEL_WITH=/path/to/metacul/models/combined_with_metadata_3b
MODEL_WITHOUT=/path/to/metacul/models/combined_without_metadata_3b
OUT_ROOT=/path/to/metacul/results/downstream_localnewsqa_pretrained_figure9_3b_correctis_qmask_full_multiseed
SLUG=correctis_qmask_a025_b125
EXPORTS="METADATA_PROMPT_STYLE=name_grounded,QA_PROMPT_STYLE=question_answer,ANSWER_CUE_STYLE=the_correct_answer_is,OMIT_OPTION_LABELS=1,EXACT_OPTION_TEXT_INSTRUCTION=0,MCQ_SCORING=option_text_avg,LENGTH_NORM_ALPHA=0.25,ADD_PROMPT_BOS=1,NULL_CALIBRATION_MODE=question_masked,NULL_CALIBRATION_BETA=1.25"

for seed in 42 43 44 45; do
  outdir="${OUT_ROOT}/seed_${seed}"
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
        --export=ALL,MODEL_TYPE=custom,TRAIN_META_TAG=${train_tag},EVAL_META_TAG=${eval_meta},META_TAG=${eval_meta},OUT_DIR=${outdir},RUN_TAG=${SLUG}_3b_${train_short}_${eval_short}_seed${seed},CUSTOM_MODEL_PATH_WITH=${MODEL_WITH},CUSTOM_MODEL_PATH_WITHOUT=${MODEL_WITHOUT},SHUFFLE_OPTIONS=1,OPTION_SHUFFLE_SEED=${seed},ANSWER_FORMAT=option,SAMPLE_SEED=${seed},NULL_QUESTION_TEXT=N/A,${EXPORTS} \
        /path/to/metacul/slurm/pretrained_localnewsqa_eval_single.slurm)
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$jid" "$seed" "$train_tag" "$eval_meta" "$SLUG" "$EXPORTS" >> "$MANIFEST"
    done
  done
done

echo "$MANIFEST"
