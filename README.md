# Metadata Localization

This repository contains the reusable software and public artifacts for the
metadata localization submission. The codebase supports three related tasks:

- building metadata-conditioned training corpora from geographically grounded
  news data;
- evaluating pretrained and instruction-tuned models on local knowledge,
  cultural projection, and external benchmark surfaces;
- reproducing the paper tables, plots, and summary analyses from tracked
  result artifacts.

The repository is organized so reviewers can quickly find the installable
software component, the benchmark data, and the experiment pipeline without
needing access to the private scratch workspace used for large raw corpora and
model checkpoints.

## Quick Start

Clone the repository and install the reusable `culture-map` package:

```bash
git clone https://github.com/iamshnoo/metadata_localization.git
cd metadata_localization
python -m pip install -e culture_map
culture-map --help
```

Render the released cultural-values map and overlay model projections:

```bash
culture-map download-paper-assets --data-dir data/paper_osf
culture-map plot-map \
  --data-dir data/paper_osf \
  --with-paper-models \
  --output outputs/culture_map.png
```

Build the tracked QA benchmark export:

```bash
python qa_data/build_hf_dataset.py --help
```

Inspect the reproducibility pipeline:

```bash
find src -maxdepth 2 -type d | sort
```

## What To Use

| Need | Start here | What it provides |
| --- | --- | --- |
| Installable software package | `culture_map/` | CLI and Python modules for WVS-style cultural projection, plotting, and provider/model evaluation. |
| Benchmark data surface | `qa_data/` | Legacy `qa_metacul` JSON, JSONL, and Hugging Face dataset export plus the dataset builder. |
| Nanotron pretraining | `src/step3_pretraining/`, `docs/PRETRAINING.md` | MAPLE Nanotron training recipes, checkpoint conversion, and pretrained-model evaluation. |
| Reproducibility pipeline | `src/` | Step-structured scripts from data processing through model evaluation and paper plots. |
| Live experiment scripts | `src_live/`, `slurm_live/` | Flat mirrors of active scripts and cluster launchers used for paper runs. |
| Paper artifacts | `results/` | Tracked summary tables, plots, benchmark outputs, and compressed large result files. |
| Workspace sync utility | `sync_from_workspace.py` | Copies curated code/results from the private workspace into this public repo layout. |

## Repository Layout

```text
metadata_localization/
├── culture_map/       # installable package: cultural map projection/evaluation
├── docs/              # reviewer-facing guides and artifact manifests
├── qa_data/           # QA benchmark data and dataset build script
├── results/           # tracked summary results, plots, and compressed artifacts
├── slurm_live/        # cluster job launchers used by the experiments
├── src/               # organized pipeline by paper stage
├── src_live/          # live flat script mirror from the experiment workspace
├── sync_from_workspace.py
└── README.md
```

The `src/` tree is the canonical organized view:

```text
src/
├── step0_dataset/          # raw NOW processing and corpus statistics
├── step1_lda_analysis/     # topic modeling and document theme scaffolding
├── step2_process_data/     # metadata, splits, and HF/MECO dataset creation
├── step3_pretraining/      # checkpoint conversion and pretraining evaluation
├── step4_sft/              # QA generation, SFT, and SFT evaluation
└── step5_plots/            # paper figures, significance, and analysis scripts
```

## Reusable Software Components

The primary reusable software component is `culture_map`, which can be installed
with `pip install -e culture_map`. It exposes the `culture-map` command and
separate modules for asset loading, scoring, projection, plotting, provider
runners, and local checkpoint evaluation.

Useful commands:

```bash
culture-map download-paper-assets
culture-map derive-projection
culture-map plot-map --with-paper-models
culture-map run-openai-part1 --model <model-name>
culture-map run-anthropic-part1 --model <model-name>
culture-map run-gemini-part1 --model <model-name>
culture-map run-together-part1 --model <model-name>
culture-map run-local-country-eval --variant <variant-name>
```

See `culture_map/README.md` and `docs/COMPONENTS.md` for the package API,
expected inputs, and common workflows.

## Reproducing Experiments

The public repository includes code and summary artifacts. Large raw corpora,
licensed inputs, model checkpoints, and cluster logs are intentionally excluded
from Git. The pipeline expects those external assets to be present in the
workspace paths configured by the individual scripts.

Recommended reading order:

1. `docs/COMPONENTS.md` for module boundaries and reusable entrypoints.
2. `docs/PRETRAINING.md` for the Nanotron training, conversion, and evaluation workflow.
3. `docs/REPRODUCIBILITY.md` for the end-to-end pipeline and external inputs.
4. `docs/RESULTS.md` for the tracked result directories and paper artifacts.
5. `qa_data/README.md` for the QA benchmark surface.
6. `culture_map/README.md` for cultural projection workflows.

## Data And Artifact Policy

The repository tracks source code, small benchmark artifacts, summary result
tables, plots, and compressed result files needed for review. It does not track:

- raw NOW corpus data;
- generated document metadata and full training corpora;
- model checkpoints and training logs;
- API keys, local environment files, or cluster-private state;
- raw result files larger than GitHub's 100 MB object limit.

One large CSV rejected by GitHub is committed as
`results/analysis/localnewsqa_locale_switch/pair_level.csv.gz`; the raw CSV is
ignored locally.

## Citation

If you use this repository, cite the accompanying submission and the benchmark
or artifact you use. A final BibTeX entry can be added here once the submission
metadata is fixed.
