# Component Guide

This guide maps the repository into reusable software components. The goal is
to make each piece understandable without reading the full experiment history.

## `culture_map/`

Installable Python package for WVS-style cultural projection and plotting.

Install:

```bash
python -m pip install -e culture_map
```

Primary command:

```bash
culture-map --help
```

Important modules:

| Module | Purpose |
| --- | --- |
| `culture_map.paper_assets` | Download and load released paper/OSF assets. |
| `culture_map.scoring` | Normalize model answers and score WVS prompt responses. |
| `culture_map.projection` | Recover the score-to-map projection used for model points. |
| `culture_map.plotting` | Render maps, overlays, and template-backed figures. |
| `culture_map.provider_runner` | Shared provider-runner utilities. |
| `culture_map.openai_runner` | OpenAI model runs for the part 1 prompt suite. |
| `culture_map.anthropic_runner` | Anthropic model runs for the part 1 prompt suite. |
| `culture_map.gemini_runner` | Gemini model runs for the part 1 prompt suite. |
| `culture_map.together_runner` | Together model runs for the part 1 prompt suite. |
| `culture_map.local_country_runner` | Local checkpoint country-targeted WVS evaluation. |

Common workflows:

```bash
culture-map download-paper-assets --data-dir data/paper_osf
culture-map derive-projection --data-dir data/paper_osf --output outputs/derived_projection.json
culture-map plot-map --data-dir data/paper_osf --with-paper-models --output outputs/map.png
```

Provider runs require the relevant API key in the environment or an env file.
The command-specific `--help` output lists the supported flags.

Local checkpoint evaluations require local model directories and the
transformers runtime used by the experiment environment. Set `MAPLE_MODEL_ROOT`
to the directory containing `combined_with_metadata_1b`,
`combined_without_metadata_1b`, `combined_with_metadata_3b`, and
`combined_without_metadata_3b` before running `run-local-country-eval`.

## `qa_data/`

Benchmark data surface for LocalNewsQA Core plus the legacy `qa_metacul`
evaluation.

### LocalNewsQA Core

`qa_data/localnewsqa_core/` is the reusable generation, validation, automated
audit, repair, and release pipeline for the final LocalNewsQA benchmark.

Useful commands:

```bash
python tools/repo.py localnewsqa-pipeline
python qa_data/localnewsqa_core/01_build_generation_requests.py --help
python qa_data/localnewsqa_core/24_audit_full_dataset_quality.py --help
python qa_data/localnewsqa_core/27_llm_verify_core_quality.py --help
```

Important files:

| Path | Purpose |
| --- | --- |
| `qa_data/localnewsqa_core/config.py` | Country, topic, quota, split, and prompt configuration. |
| `qa_data/localnewsqa_core/prompts/` | Developer prompts for explicit and ambiguous candidate generation. |
| `qa_data/localnewsqa_core/human_validation_guidelines.md` | Reviewer-facing validation rubric. |
| `qa_data/localnewsqa_core/final_gold_20260516/` | Final 18,700 row JSONL release and summary. |
| `qa_data/hf_dataset_localnewsqa_gold_20260516/` | Compact Hugging Face-style parquet export. |

The final release has 17,000 explicit rows and 1,700 ambiguous rows balanced
across 17 countries. See `docs/DATASET_AND_AUDIT.md` for the full stage map.

### Legacy `qa_metacul`

The older benchmark is preserved for compatibility with earlier runs. It
includes:

- continent-level JSON inputs;
- `hf_dataset.jsonl`, a normalized JSONL export;
- `hf_dataset/`, a saved Hugging Face dataset directory;
- `build_hf_dataset.py`, the builder/normalizer;
- prompt templates used to generate the benchmark.

Run:

```bash
python qa_data/build_hf_dataset.py --help
```

The same builder is mirrored under `src/step4_sft/4a_qa_data_generation/` so
the paper pipeline can refer to it from the staged source tree.

## `src/`

Canonical, organized experiment pipeline. Each numbered directory corresponds
to a stage of the submission workflow.

| Directory | Role |
| --- | --- |
| `step0_dataset/` | Raw NOW data preparation and corpus statistics. |
| `step1_lda_analysis/` | Topic-modeling and document theme scaffolding. |
| `step2_process_data/` | Metadata creation, split definitions, and HF/MECO dataset builders. |
| `step3_pretraining/` | Nanotron run recipes, checkpoint conversion, perplexity lists, and pretraining analyses. |
| `step4_sft/` | QA generation, SFT training, LoRA merge, and SFT evaluation. |
| `step5_plots/` | Plot generation, significance tests, and summary analyses. |

Most scripts are designed as reproducibility entrypoints rather than library
APIs. Use the directory names and numbered filenames to follow the paper order.

### Nanotron Pretraining

`src/step3_pretraining/3a_pretrain/continents/` contains copy-pastable
Nanotron `slurm_launcher.py` commands for 500M and 1B MAPLE runs, including
with-metadata, without-metadata, leave-one-continent, and metadata-field
ablations. The training engine is the project Nanotron fork at
`https://github.com/iamshnoo/nanotron`; this repository stores the MAPLE
recipes and downstream conversion/evaluation utilities. See
`docs/PRETRAINING.md` for the full usage guide.

### SFT

`src/step4_sft/4b_sft/` contains the supervised fine-tuning component. The
entrypoints train with-metadata and without-metadata adapters, merge LoRA
weights back into model checkpoints, and preserve the SLURM manifests used for
the paper runs.

```bash
python tools/repo.py sft
```

Start with:

| File | Purpose |
| --- | --- |
| `src/step4_sft/4b_sft/12_sft.py` | Earlier SFT training entrypoint. |
| `src/step4_sft/4b_sft/15_sft.py` | Later SFT training entrypoint used by the final runs. |
| `src/step4_sft/4b_sft/13_merge_lora.py` | Earlier adapter merge entrypoint. |
| `src/step4_sft/4b_sft/16_merge_lora.py` | Later adapter merge entrypoint used by the final runs. |
| `src/step4_sft/4b_sft/scripts/` | Cluster launch manifests. |

### Evaluations

The evaluation surface spans pretraining, SFT, LocalNewsQA, external
benchmarks, cultural projection, open-ended geographic probes, significance,
and plots.

```bash
python tools/repo.py evals
```

Start with:

| Area | Entry points |
| --- | --- |
| Pretraining perplexity | `src/step3_pretraining/3b_pretrain_eval/` |
| SFT/local QA evaluation | `src/step4_sft/4c_sft_eval/` |
| Cultural projection | `culture_map/`, `slurm_live/submit_culture_map_wvs_maple.sh` |
| Live benchmark runs | `slurm_live/pretrained_localnewsqa_eval_*.slurm`, `slurm_live/pretrained_external_eval_*.slurm` |
| Plots and significance | `src/step5_plots/`, `results/analysis/` |

See `docs/SFT_AND_EVALUATION.md` for the full map.

## `tools/`

Python helper entrypoints. `tools/repo.py` is intentionally small and
dependency-free; it records direct Python discovery commands:

```bash
python tools/repo.py components
python tools/repo.py pretraining-recipes
python tools/repo.py localnewsqa-pipeline
python tools/repo.py sft
python tools/repo.py evals
```

## `src_live/`

Flat mirror of the live experiment-workspace scripts. This is useful when
matching cluster logs, historical commands, or `slurm_live/` job files that
reference the original script names. New users should prefer the organized
`src/` tree unless they are reproducing an exact live run.

## `slurm_live/`

SLURM launchers used for model conversion, evaluation, plotting, benchmark
reruns, and result post-processing. These scripts are cluster-specific, but they
document the exact command arguments used for paper runs.

## `results/`

Tracked result artifacts. This directory is intentionally summary-heavy:
tables, plots, JSON summaries, benchmark score files, and compressed large
artifacts. It does not contain checkpoints, raw corpora, or full logs.

See `docs/RESULTS.md` for a directory-by-directory inventory.

## `sync_from_workspace.py`

Repository curation utility. It copies selected code, benchmark files, and
summary artifacts from the experiment workspace into this repository layout. It
is useful for maintainers, not required for running the public components.
