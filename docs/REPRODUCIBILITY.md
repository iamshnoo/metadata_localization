# Reproducibility Guide

This repository separates reusable public code from large external assets. The
scripts are reproducible when the required corpora, checkpoints, and cluster
environment are available, but the public Git repository intentionally does not
store those heavyweight inputs.

## External Inputs

Required outside Git:

- raw English News on the Web corpus files;
- generated document metadata and processed training corpora;
- trained model checkpoints and LoRA adapters;
- LocalNewsQA intermediate `runs/` artifacts, raw provider outputs, and web
  fetch caches;
- API credentials for hosted model providers;
- cluster-specific modules, partitions, cache paths, and job logs.

Ignored directories include `data/`, `document_metadata/`, `training_data/`,
`models/`, `logs/`, and `latex/`.

## Pipeline Order

### 1. Corpus and metadata preparation

```text
src/step0_dataset/
src/step1_lda_analysis/
src/step2_process_data/2a_metadata_processing/
```

These scripts prepare raw documents, fit topic models, create document metadata,
and build metadata indices. They expect the corpus workspace to be available.

### 2. Dataset construction

```text
src/step2_process_data/2b_hf_dataset_creation/
```

This stage creates Hugging Face and MECO-style datasets, metadata ablations,
combined splits, and leave-one-out variants. The companion SLURM scripts under
the same directory show the cluster entrypoints.

### 3. Pretraining and checkpoint conversion

```text
src/step3_pretraining/
```

This stage contains Nanotron training recipes, checkpoint conversion helpers,
and pretraining evaluation scripts. The training engine is the project fork at
`https://github.com/iamshnoo/nanotron`; this repository stores the MAPLE run
manifests and downstream tooling. Large checkpoints live outside Git; the
repository tracks only recipes, conversion/evaluation code, and selected summary
outputs. See `docs/PRETRAINING.md`.

### 4. LocalNewsQA generation and audit

```text
qa_data/localnewsqa_core/
qa_data/hf_dataset_localnewsqa_gold_20260516/
```

This stage builds LocalNewsQA Core candidates, runs human-validation and web
evidence workflows, applies automated heuristic/semantic/LLM audits, repairs
weak rows, and emits the final gold release. The repository tracks the complete
script surface, prompts, guidelines, final JSONL release, final summary, and
parquet export. It excludes the multi-GB `runs/` workspace.

Useful discovery command:

```bash
python tools/repo.py localnewsqa-pipeline
```

See `docs/DATASET_AND_AUDIT.md`.

### 5. SFT training and adapter merge

```text
src/step4_sft/4a_qa_data_generation/
src/step4_sft/4b_sft/
```

The legacy QA benchmark builder is available directly from `qa_data/` and
mirrored inside `src/step4_sft/4a_qa_data_generation/`. SFT scripts train
adapters for metadata variants and merge LoRA weights into checkpoints for
evaluation.

```bash
python tools/repo.py sft
```

See `docs/SFT_AND_EVALUATION.md`.

### 6. Evaluation, analysis, and paper artifacts

```text
src/step3_pretraining/3b_pretrain_eval/
src/step4_sft/4c_sft_eval/
src/step5_plots/
slurm_live/
results/
```

Evaluation covers pretraining perplexity, LocalNewsQA, external benchmarks,
SFT comparisons, WVS/culture-map projections, open-ended geographic probes,
significance tests, and paper plots. Plotting and analysis scripts consume
tracked result summaries plus external artifacts when needed. Most paper-facing
tables, plots, and summaries are stored under `results/`.

```bash
python tools/repo.py evals
```

## Reusable Package Workflow

The `culture_map` package can be used without the full training workspace:

```bash
python -m pip install -e culture_map
culture-map download-paper-assets --data-dir data/paper_osf
culture-map derive-projection --data-dir data/paper_osf
culture-map plot-map --data-dir data/paper_osf --with-paper-models
```

Hosted provider evaluations require provider-specific API keys. Local checkpoint
evaluations require local model paths and the tokenizer/model runtime used by
the experiment environment.

## Cluster Workflow

The files in `slurm_live/` are preserved as executable run manifests. They are
not intended to be portable without editing account, partition, cache, and model
paths. Their value is to document the exact orchestration used for the
submission runs.

## Python Entry Point Index

This repository uses direct Python commands for component discovery:

```bash
python tools/repo.py components
python tools/repo.py pretraining-recipes
python tools/repo.py localnewsqa-pipeline
python tools/repo.py sft
python tools/repo.py evals
```
