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

Benchmark data surface for the legacy `qa_metacul` evaluation. It includes:

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
| `step3_pretraining/` | Checkpoint conversion, perplexity lists, and pretraining analyses. |
| `step4_sft/` | QA generation, SFT training, LoRA merge, and SFT evaluation. |
| `step5_plots/` | Plot generation, significance tests, and summary analyses. |

Most scripts are designed as reproducibility entrypoints rather than library
APIs. Use the directory names and numbered filenames to follow the paper order.

## `src_live/`

Flat mirror of the live scratch-workspace scripts. This is useful when matching
cluster logs, historical commands, or `slurm_live/` job files that reference the
original script names. New users should prefer the organized `src/` tree unless
they are reproducing an exact live run.

## `slurm_live/`

SLURM launchers used for model conversion, evaluation, plotting, benchmark
reruns, and result post-processing. These scripts are cluster-specific, but they
document the exact command arguments used for paper runs.

## `results/`

Tracked result artifacts. This directory is intentionally summary-heavy:
tables, plots, JSON summaries, benchmark score files, and compressed large
artifacts. It does not contain private checkpoints, raw corpora, or full logs.

See `docs/RESULTS.md` for a directory-by-directory inventory.

## `sync_from_workspace.py`

Repository curation utility. It copies selected code, benchmark files, and
summary artifacts from the private scratch workspace into this public repository
layout. It is useful for maintainers, not required for reviewers.
