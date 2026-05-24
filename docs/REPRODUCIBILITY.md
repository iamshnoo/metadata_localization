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
and build metadata indices. They expect the private corpus workspace to be
available.

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

This stage contains checkpoint conversion helpers and pretraining evaluation
scripts. Large checkpoints live outside Git; the repository tracks only the
conversion/evaluation code and selected summary outputs.

### 4. QA generation, SFT, and evaluation

```text
src/step4_sft/
qa_data/
```

The QA benchmark builder is available directly from `qa_data/` and mirrored
inside `src/step4_sft/4a_qa_data_generation/`. SFT scripts train adapters,
merge LoRA weights, and evaluate metadata-sensitive behavior.

### 5. Analysis and paper artifacts

```text
src/step5_plots/
results/
```

Plotting and analysis scripts consume tracked result summaries plus external
artifacts when needed. Most paper-facing tables, plots, and summaries are stored
under `results/`.

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

## Large Result Handling

GitHub rejects individual files larger than 100 MB. Large raw outputs are either
excluded, summarized, or compressed. The compressed file
`results/analysis/localnewsqa_locale_switch/pair_level.csv.gz` replaces the raw
CSV that exceeded GitHub's object size limit.
