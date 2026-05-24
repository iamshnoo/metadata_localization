# Results Inventory

The `results/` directory stores reviewable artifacts from the submission:
summary tables, score files, plots, and compressed large outputs. It is not a
complete experiment cache.

## Top-Level Summary Files

| File | Contents |
| --- | --- |
| `results/perplexity_eval*.csv` | Perplexity summaries across model/data variants. |
| `results/qa_metacul_eval*.csv` | Legacy QA benchmark evaluation summaries. |
| `results/qa_metacul_eval_adversarial*.csv` | Adversarial/localization QA analyses. |
| `results/hf_collection_sync_report*.json` | Hugging Face collection sync reports. |

## Main Directories

| Directory | Contents |
| --- | --- |
| `results/analysis/` | Aggregated analysis CSV/JSON files, protocol summaries, bootstrap/significance outputs, and compressed large pair-level data. |
| `results/appendix_model_gain_tables_20260505*/` | Appendix gain-table sources and rendered LaTeX rows. |
| `results/culture_map_wvs_*` | WVS cultural projection outputs, country-distance summaries, and model response tables. |
| `results/downstream*` | Downstream LocalNewsQA and QA evaluation JSONL outputs. |
| `results/external_benchmarks*` | External benchmark score files and protocol follow-up results. |
| `results/final_benchmark_matrix/` | Consolidated benchmark matrix artifacts. |
| `results/localnewsqa_gold_20260516/` | LocalNewsQA gold-set evaluation summaries and related outputs. |
| `results/mechanistic/` | Mechanistic and lexical probe summaries. |
| `results/noamerica_cluster/` | Leave-one-region cluster analyses. |
| `results/perplexity/` | Perplexity evaluation lists and derived summaries. |
| `results/plots/` | Paper and appendix plot sources and rendered figures. |
| `results/significance/` | Statistical tests and confidence interval summaries. |
| `results/url_analysis/` | URL-signal and adversarial metadata analyses. |

## Relationship To Code

- `qa_data/localnewsqa_core/final_gold_20260516/` contains the final gold
  benchmark data evaluated by the LocalNewsQA result summaries.
- `docs/DATASET_AND_AUDIT.md` describes the generation and quality-audit
  pipeline behind that benchmark.
- `docs/SFT_AND_EVALUATION.md` maps the SFT, benchmark, and significance
  scripts that consume these result files.
- `src/step5_plots/` contains the plotting and table-generation scripts.
- `src_live/` contains live summary scripts used for many final result sweeps.
- `slurm_live/` contains the run manifests for cluster-generated outputs.
- `culture_map/` can regenerate WVS map overlays and country-projection tables
  when the required model/provider inputs are available.
