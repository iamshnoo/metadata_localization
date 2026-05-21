#!/usr/bin/env bash
set -euo pipefail

ROOT=/path/to/metacul
PYTHON=${PYTHON:-/path/to/nanotron-b200/bin/python}
RUN_ROOT=${RUN_ROOT:-$ROOT/results/localnewsqa_gold_20260516}
LOG_PREFIX="[lnqa-gold-post]"
WAIT_SECONDS=${WAIT_SECONDS:-172800}
POLL_SECONDS=${POLL_SECONDS:-300}

required_files=(
  "$RUN_ROOT/plots/plot_8_pretrained_target_split_multiseed.csv"
  "$RUN_ROOT/plots/plot_8_pretrained_target_split_seed41_bootstrap.csv"
  "$RUN_ROOT/summaries/localnewsqa_locale_switch/summary.csv"
  "$RUN_ROOT/plots/plot_8_sft_target_split_multiseed_gold.csv"
  "$RUN_ROOT/summaries/localnewsqa_sft_locale_switch/summary.csv"
  "$RUN_ROOT/plots/table7_external_localnewsqa_multiseed_long.csv"
  "$RUN_ROOT/summaries/localnewsqa_external_locale_switch/summary.csv"
  "$RUN_ROOT/plots/adversarial_pretrained_summary_full_mismatch.csv"
  "$RUN_ROOT/plots/adversarial_pretrained_summary_urlmask.csv"
)

deadline=$((SECONDS + WAIT_SECONDS))
while true; do
  missing=()
  for path in "${required_files[@]}"; do
    if [[ ! -s "$path" ]]; then
      missing+=("$path")
    fi
  done
  if ((${#missing[@]} == 0)); then
    break
  fi
  if ((SECONDS >= deadline)); then
    printf '%s timed out waiting for summary files:\n' "$LOG_PREFIX" >&2
    printf '  %s\n' "${missing[@]}" >&2
    exit 1
  fi
  printf '%s waiting for %s summary files; sleeping %ss\n' "$LOG_PREFIX" "${#missing[@]}" "$POLL_SECONDS"
  sleep "$POLL_SECONDS"
done

printf '%s building LocalNewsQA gold gain table\n' "$LOG_PREFIX"
"$PYTHON" "$ROOT/src/72_localnewsqa_gold_model_gain_tables.py" --strict

printf '%s writing seed-41 alpha/beta harness-search audits\n' "$LOG_PREFIX"
mkdir -p "$RUN_ROOT/summaries/harness_search"
"$PYTHON" "$ROOT/src/48_search_figure9_scores.py" \
  --root "$RUN_ROOT/pretrained_target/1b_codeg_labels_question_final/seed_41" \
  > "$RUN_ROOT/summaries/harness_search/pretrained_1b_seed41_alpha_beta.jsonl"
"$PYTHON" "$ROOT/src/48_search_figure9_scores.py" \
  --root "$RUN_ROOT/pretrained_target/3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41" \
  > "$RUN_ROOT/summaries/harness_search/pretrained_3b_seed41_alpha_beta.jsonl"
"$PYTHON" "$ROOT/src/48_search_figure9_scores.py" \
  --root "$RUN_ROOT/sft_target/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos/seed_41" \
  > "$RUN_ROOT/summaries/harness_search/sft_1b_seed41_alpha_beta.jsonl"
"$PYTHON" "$ROOT/src/48_search_figure9_scores.py" \
  --root "$RUN_ROOT/sft_target/3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41" \
  > "$RUN_ROOT/summaries/harness_search/sft_3b_seed41_alpha_beta.jsonl"

printf '%s rendering main LocalNewsQA figure\n' "$LOG_PREFIX"
"$PYTHON" "$ROOT/src/73_plot_localnewsqa_gold_accuracy_switch.py" --write-latex

printf '%s rendering SFT appendix figure\n' "$LOG_PREFIX"
"$PYTHON" "$ROOT/src/60_plot_localnewsqa_sft_clean.py" \
  --csv-path "$RUN_ROOT/plots/plot_8_sft_target_split_multiseed_gold.csv" \
  --out-results-pdf "$RUN_ROOT/plots/18_sft_localnewsqa_accuracy_apples_to_apples.pdf" \
  --out-results-png "$RUN_ROOT/plots/18_sft_localnewsqa_accuracy_apples_to_apples.png" \
  --out-latex-pdf "$ROOT/latex/figs/appendix/18_sft_localnewsqa_accuracy_apples_to_apples.pdf"

printf '%s rendering country-level SFT appendix figures\n' "$LOG_PREFIX"
"$PYTHON" "$ROOT/src/76_plot_localnewsqa_gold_country_accuracy.py" --strict --write-latex

printf '%s rendering adversarial appendix figures\n' "$LOG_PREFIX"
"$PYTHON" "$ROOT/src/77_render_localnewsqa_gold_adversarial_plots.py"

printf '%s checking win conditions\n' "$LOG_PREFIX"
"$PYTHON" "$ROOT/src/74_check_localnewsqa_gold_win_conditions.py"

printf '%s compiling paper PDF\n' "$LOG_PREFIX"
cd "$ROOT/latex"
latexmk -pdf -interaction=nonstopmode -halt-on-error arr_may26.tex

printf '%s done\n' "$LOG_PREFIX"
