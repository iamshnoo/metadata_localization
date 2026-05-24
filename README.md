# Metadata Localization

This repository contains the reusable software and public artifacts for the
metadata localization submission. The codebase supports the full component
surface used by the project:

- building metadata-conditioned training corpora from geographically grounded
  news data;
- running MAPLE pretraining recipes through the project Nanotron fork;
- generating, validating, auditing, and releasing LocalNewsQA benchmark data;
- supervised fine-tuning and LoRA merge workflows;
- evaluating pretrained and instruction-tuned models on local knowledge,
  cultural projection, and external benchmark surfaces;
- reproducing the paper tables, plots, and summary analyses from tracked
  result artifacts.

The top-level layout separates the reusable package, benchmark data, training
recipes, evaluation scripts, and paper artifacts so each component can be used
directly.

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
python tools/repo.py components
python tools/repo.py localnewsqa-pipeline
python tools/repo.py sft
python tools/repo.py evals
```

## What To Use

| Need | Start here | What it provides |
| --- | --- | --- |
| Installable software package | `culture_map/` | CLI and Python modules for WVS-style cultural projection, plotting, and provider/model evaluation. |
| LocalNewsQA gold benchmark | `qa_data/localnewsqa_core/final_gold_20260516/`, `qa_data/hf_dataset_localnewsqa_gold_20260516/` | Final 18,700 row explicit/ambiguous gold benchmark as JSONL and parquet. |
| Dataset generation and audit | `qa_data/localnewsqa_core/`, `docs/DATASET_AND_AUDIT.md` | LocalNewsQA candidate generation, human validation, web evidence, automated audit, repair, and release scripts. |
| Legacy QA benchmark | `qa_data/` | Legacy `qa_metacul` JSON, JSONL, Hugging Face dataset export, and builder. |
| Nanotron pretraining | `src/step3_pretraining/`, `docs/PRETRAINING.md` | MAPLE Nanotron training recipes, checkpoint conversion, and pretrained-model evaluation. |
| SFT training and merge | `src/step4_sft/4b_sft/`, `docs/SFT_AND_EVALUATION.md` | Supervised fine-tuning, LoRA merge, and cluster launch manifests for metadata variants. |
| Evaluation stack | `src/step3_pretraining/3b_pretrain_eval/`, `src/step4_sft/4c_sft_eval/`, `slurm_live/`, `docs/SFT_AND_EVALUATION.md` | LocalNewsQA, external benchmarks, WVS/culture-map, open-ended probes, significance, and plots. |
| Reproducibility pipeline | `src/` | Step-structured scripts from data processing through model evaluation and paper plots. |
| Live experiment scripts | `src_live/`, `slurm_live/` | Flat mirrors of active scripts and cluster launchers used for paper runs. |
| Paper artifacts | `results/` | Tracked summary tables, plots, benchmark outputs, and compressed large result files. |
| Python component index | `tools/repo.py` | Small no-dependency helper for discovering component entrypoints. |
| Workspace sync utility | `sync_from_workspace.py` | Copies curated code/results from the experiment workspace into this repo layout. |

## Repository Layout

```text
metadata_localization/
├── culture_map/       # installable package: cultural map projection/evaluation
├── docs/              # component guides and artifact manifests
├── qa_data/           # QA benchmark data, LocalNewsQA Core, and builders
├── results/           # tracked summary results, plots, and compressed artifacts
├── slurm_live/        # cluster job launchers used by the experiments
├── src/               # organized pipeline by paper stage
├── src_live/          # live flat script mirror from the experiment workspace
├── tools/             # Python helper entrypoints for repository discovery
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

Use `python tools/repo.py components` to list the main reusable entrypoints.

`culture_map` is an installable package we use for WVS-style cultural
projection. Install it with `pip install -e culture_map`; it exposes the
`culture-map` command and separate modules for asset loading, scoring,
projection, plotting, provider runners, and local checkpoint evaluation.

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

The other reusable components are script-level pipelines:

- `qa_data/localnewsqa_core/` for LocalNewsQA generation, validation, automated
  audit, and final release construction.
- `src/step3_pretraining/` for MAPLE pretraining recipes using
  `https://github.com/iamshnoo/nanotron`, checkpoint conversion, and pretrained
  evaluation.
- `src/step4_sft/4b_sft/` for SFT training and LoRA merge.
- `src/step4_sft/4c_sft_eval/` and `src/step5_plots/` for benchmark
  evaluation, significance testing, and paper plots.

## Reproducing Experiments

The repository includes code and summary artifacts. Large raw corpora,
licensed inputs, model checkpoints, and cluster logs are intentionally excluded
from Git. The pipeline expects those external assets to be present in the
workspace paths configured by the individual scripts.

Recommended reading order:

1. `docs/COMPONENTS.md` for module boundaries and reusable entrypoints.
2. `docs/PRETRAINING.md` for the Nanotron training, conversion, and evaluation workflow.
3. `docs/DATASET_AND_AUDIT.md` for LocalNewsQA generation and audit.
4. `docs/SFT_AND_EVALUATION.md` for SFT and benchmark evaluation.
5. `docs/REPRODUCIBILITY.md` for the end-to-end pipeline and external inputs.
6. `docs/RESULTS.md` for the tracked result directories and paper artifacts.
7. `qa_data/README.md` for the QA benchmark surface.
8. `culture_map/README.md` for cultural projection workflows.

## Data And Artifact Policy

The repository tracks source code, small benchmark artifacts, summary result
tables, plots, and compressed result files needed for review. It does not track:

- raw NOW corpus data;
- generated document metadata and full training corpora;
- model checkpoints and training logs;
- API keys, local environment files, or cluster-local state;
- raw result files larger than GitHub's 100 MB object limit.

One large CSV rejected by GitHub is committed as
`results/analysis/localnewsqa_locale_switch/pair_level.csv.gz`; the raw CSV is
ignored locally.

## Citation

If you use this repository, cite the accompanying submission and the benchmark
or artifact you use. A final BibTeX entry can be added here once the submission
metadata is fixed.
